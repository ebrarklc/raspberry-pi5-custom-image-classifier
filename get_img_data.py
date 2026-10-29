#!/usr/bin/env python3
"""
get_img_data.py: Raspberry Pi kamerasıyla etiketli veri seti toplama aracı
Kullanım: python get_img_data.py  →  tarayıcıda http://<PI_IP>:5000
"""
import os
import time
import threading
import cv2
from flask import Flask, Response, request, redirect, url_for, render_template_string
from picamera2 import Picamera2

BASE_DIR = "dataset"     # görüntüler dataset/<etiket>/ klasörlerine kaydedilir

# ── PARÇA 1: Kamerayı aç ve ayarla ──────────────────────────────
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"},   # ISP 320x240 verir, numpy'da [R,G,B]
    sensor={"output_size": (1640, 1232)}              # tam görüş alanı
)
picam2.configure(config)
picam2.start()
time.sleep(2)                                         # otomatik pozlama otursun

# ── PARÇA 2: Arka planda sürekli kare al ────────────────────────
latest_jpeg = None                # thread'ler arası paylaşılan "son kare kutusu"
lock = threading.Lock()           # kutunun kilidi
current_label = None              # şu an hangi sınıfı çekiyoruz?


def camera_loop():
    """Arka planda durmadan: kare al → JPEG'e çevir → kutuya koy"""
    global latest_jpeg
    while True:
        frame = picam2.capture_array()                    # RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)      # OpenCV BGR sırası ister
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            with lock:                                    # 🔒 anahtarı al
                latest_jpeg = buf.tobytes()               # kutuya koy
        time.sleep(0.03)                                  # ~30 kare/saniye


# ── YARDIMCI FONKSİYONLAR ───────────────────────────────────────
def count_images(label):
    """Bir sınıf klasöründeki .jpg sayısı (script yeniden başlasa bile doğru sayar)"""
    folder = os.path.join(BASE_DIR, label)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


def all_counts():
    """Tüm sınıflar ve görüntü sayıları: {'kamera': 60, ...}"""
    return {d: count_images(d) for d in sorted(os.listdir(BASE_DIR))
            if os.path.isdir(os.path.join(BASE_DIR, d))}


# ── HTML ŞABLONLARI ─────────────────────────────────────────────
INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Veri Toplama</title>
<style>body{font-family:sans-serif;max-width:720px;margin:24px auto}
input,button{font-size:18px;padding:8px 14px} li{font-size:18px;margin:6px 0}</style></head>
<body>
  <h1>📸 Veri Seti Toplama</h1>
  <form method="post">
    <input name="label" placeholder="etiket (ör. kamera)" required
           pattern="[a-z0-9_]+" title="Sadece küçük harf, rakam ve _">
    <button>Başla ▶</button>
  </form>
  <h3>Mevcut sınıflar</h3>
  <ul>
  {% for label, n in counts.items() %}
    <li><a href="{{ url_for('select', label=label) }}">{{ label }}</a>: <b>{{ n }}</b> görüntü</li>
  {% else %}
    <li>Henüz sınıf yok</li>
  {% endfor %}
  </ul>
</body></html>
"""

CAPTURE_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>{{ label }}</title>
<style>body{font-family:sans-serif;max-width:720px;margin:24px auto;transition:background .15s}
button{font-size:20px;padding:10px 18px}</style></head>
<body>
  <h1>Sınıf: <code>{{ label }}</code></h1>
  <p style="font-size:20px">Bu sınıfta <b id="n">{{ n }}</b> görüntü var</p>
  <img src="{{ url_for('stream') }}" width="640" height="480" style="border:2px solid #333">
  <p><button onclick="snap()">📸 Çek (Space)</button> &nbsp; <a href="/">↩ Sınıf değiştir</a></p>
<script>
function snap() {
  fetch('/snap', {method: 'POST'})                       // arka planda sunucuya "çek" de
    .then(r => r.json())
    .then(d => {
      document.getElementById('n').textContent = d.count;           // sayacı güncelle
      document.body.style.background = '#c8f7c5';                   // yeşil flaş = kaydedildi
      setTimeout(() => document.body.style.background = '', 150);
    });
}
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); snap(); }           // Space = fotoğraf çek
});
</script>
</body></html>
"""

# ── PARÇA 3 + 4: Web sunucusu ve rotalar ────────────────────────
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """GET: etiket sayfasını göster | POST: yeni etiketi kaydet, çekim sayfasına git"""
    global current_label
    if request.method == "POST":
        current_label = request.form["label"].strip().lower()
        os.makedirs(os.path.join(BASE_DIR, current_label), exist_ok=True)
        return redirect(url_for("capture"))
    return render_template_string(INDEX_HTML, counts=all_counts())


@app.route("/select/<label>")
def select(label):
    """Listedeki var olan bir sınıfa tıklayınca"""
    global current_label
    current_label = label
    return redirect(url_for("capture"))


@app.route("/capture")
def capture():
    """Çekim sayfası: canlı görüntü + sayaç + çek butonu"""
    if current_label is None:
        return redirect(url_for("index"))
    return render_template_string(CAPTURE_HTML, label=current_label, n=count_images(current_label))


@app.route("/stream")
def stream():
    """MJPEG akışı: kutudaki son kareyi durmadan tarayıcıya gönder"""
    def generate():
        while True:
            with lock:
                jpg = latest_jpeg
            if jpg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snap", methods=["POST"])
def snap():
    """Kutudaki son kareyi (ekranda gördüğünü) diske kaydet"""
    if current_label is None:
        return {"error": "Önce bir etiket seç"}, 400
    with lock:
        jpg = latest_jpeg
    ms = int(time.time() * 1000) % 1000                   # milisaniye → isim çakışmasın
    filename = f"{current_label}_{time.strftime('%Y%m%d-%H%M%S')}-{ms:03d}.jpg"
    with open(os.path.join(BASE_DIR, current_label, filename), "wb") as f:
        f.write(jpg)
    return {"count": count_images(current_label)}         # JSON cevap


# ── ANA PROGRAM ─────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    threading.Thread(target=camera_loop, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        picam2.stop()
        picam2.close()
        print("\n📊 Özet:", all_counts())
