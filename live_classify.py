"""live_classify.py: int8 modelle canlı sınıflandırma. Kullanım: python live_classify.py → http://<PI_IP>:5000"""
import time
import threading
import numpy as np
import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter

LABELS = ["arka_plan", "kamera", "pi_kutusu"]          # alfabetik sıra!
THRESHOLD = 0.80                                       # bunun altı = "belirsiz"
COLORS = {"arka_plan": (200, 200, 200), "kamera": (0, 200, 0), "pi_kutusu": (0, 0, 255)}  # BGR

# ── Model: bir kez yükle ─────────────────────────────────────────
itp = Interpreter(model_path="models/model_int8.tflite", num_threads=4)
itp.allocate_tensors()
inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
s_out, z_out = out['quantization']

# ── Kamera: veri toplarken kullandığımız AYNI ayarlar ────────────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"},
    sensor={"output_size": (1640, 1232)}))
picam2.start()
time.sleep(2)

latest_jpeg = None
lock = threading.Lock()


def classify(frame):
    x = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_AREA)   # squash
    x = (x.astype(np.int16) - 128).astype(np.int8)[None]              # int8 girişi
    itp.set_tensor(inp['index'], x)
    itp.invoke()
    return (itp.get_tensor(out['index'])[0].astype(np.float32) - z_out) * s_out


def loop():
    global latest_jpeg
    fps, t_prev = 0.0, time.perf_counter()
    while True:
        frame = picam2.capture_array()                                 # RGB
        t0 = time.perf_counter()
        probs = classify(frame)
        ms = (time.perf_counter() - t0) * 1000
        i = int(np.argmax(probs))
        label, conf = LABELS[i], probs[i]

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - t_prev)                         # yumuşatılmış FPS
        t_prev = now

        if conf >= THRESHOLD:
            text, color = f"{label} {conf*100:.0f}%", COLORS[label]
        else:
            text, color = f"belirsiz ({label} {conf*100:.0f}%)", (0, 165, 255)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (320, 44), (0, 0, 0), -1)
        cv2.putText(bgr, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(bgr, f"cikarim {ms:.1f} ms | {fps:.0f} FPS", (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", bgr)
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()


app = Flask(__name__)


@app.route("/")
def index():
    return '<h1>🎥 Canlı Sınıflandırma</h1><img src="/stream" width="640" height="480">'


@app.route("/stream")
def stream():
    def gen():
        while True:
            with lock:
                jpg = latest_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        picam2.stop()
        picam2.close()
