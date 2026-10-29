# Bölüm 3 — 🏆 Proje 1: Kendi Görüntü Sınıflandırıcın

> **Kaynak:** Marcelo Rovai, *Edge AI Engineering — Hands-On with the Raspberry Pi*, s. 110–144
> **Ön koşul:** [Bölüm 2 — Görüntü sınıflandırma temelleri](https://github.com/ebrarklc/raspberry-pi5-image-classification-litert) (venv `~/tflite_env`, LiteRT)
> **Teori:** [Görüntü sınıflandırma teorisi](https://github.com/ebrarklc/raspberry-pi5-image-classification-litert/blob/main/image-classification-teorisi.md) — özellikle Bölüm 12 (veri), 13 (overfitting), 15 (transfer learning)

---

## İçindekiler
1. [Proje akışı](#1-proje-akışı)
2. [P1.1 — Hedef ve sınıflar](#2-p11--hedef-ve-sınıflar)
3. [P1.2 — Veri toplama aracını adım adım yazmak](#3-p12--veri-toplama-aracını-adım-adım-yazmak)
4. [P1.3 — Veri toplama](#4-p13--veri-toplama)
5. [P1.4 — Edge Impulse ile eğitim](#5-p14--edge-impulse-ile-eğitim)
6. [P1.5 — Test](#6-p15--test)
7. [P1.6 — Modeli Pi'ye almak ve incelemek](#7-p16--modeli-pye-almak-ve-incelemek)
8. [P1.7 — Canlı sınıflandırma](#8-p17--canlı-sınıflandırma)
9. [Yaşanan hatalar ve dersleri](#9-yaşanan-hatalar-ve-dersleri)
10. [Terimler sözlüğü](#10-terimler-sözlüğü)
11. [Mülakat notları](#11-mülakat-notları)

---

## 1. Proje akışı

```
🎯 Hedef → 📸 Veri seti → 🔧 Ön işleme → 🧠 Eğitim → 🧪 Test → 🚀 Dağıtım
           (Pi)          (Edge Impulse — bulut)                  (Pi)
```

**Neden eğitim bulutta?** Eğitim = ileri geçiş + geri geçiş + ağırlık güncelleme, binlerce kez. Pi'nin CPU'sunda saatler sürer; bulut GPU'sunda dakikalar. Pi yalnızca **çıkarım** yapar.

---

## 2. P1.1 — Hedef ve sınıflar

| Sınıf | Açıklama |
|---|---|
| `kamera` | Yedek Raspberry Pi Camera Module v2 |
| `pi_kutusu` | Raspberry Pi'nin kırmızı-beyaz kutusu |
| `arka_plan` | Nesneler yokken **aynı** çalışma alanı |

**Kurallar:**
- Etiketlerde Türkçe karakter ve boşluk yok (`a-z`, `0-9`, `_`).
- `arka_plan` sınıfı zorunlu: yoksa model önü boşken de iki nesneden birini seçmek zorunda kalır (**closed-set** problemi).
- Seçilen nesneler ImageNet'te yok → transfer learning'in gücünü gösterir.

---

## 3. P1.2 — Veri toplama aracını adım adım yazmak

**Neden kendi aracımız?** 180 fotoğraf için terminal komutu yazmak yerine: tarayıcıda canlı görüntü + **Space** ile çekim + otomatik doğru klasöre kayıt.

**Yöntem:** Kodu **parça parça yaz, her parçayı test et.** Hata çıkarsa hep son eklenen parçadadır.

### Hazırlık
```bash
source ~/tflite_env/bin/activate
pip install flask
mkdir -p ~/Documents/TFLITE/CUSTOM_CLASS
cd ~/Documents/TFLITE/CUSTOM_CLASS
```

### Parça 1 — Kamerayı doğru ayarlarla açmak
```python
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"},   # ISP doğrudan QVGA verir; numpy'da [R,G,B]
    sensor={"output_size": (1640, 1232)}              # tam görüş alanı (kırpılmış mod değil)
)
picam2.configure(config); picam2.start(); time.sleep(2)
frame = picam2.capture_array()
print(frame.shape)    # (240, 320, 3) → (yükseklik, genişlik, kanal)
```

### Parça 2 — Arka planda sürekli kare almak (thread + lock)
```python
latest_jpeg = None
lock = threading.Lock()

def camera_loop():
    global latest_jpeg
    while True:
        frame = picam2.capture_array()                                  # RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                    # OpenCV BGR ister
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()
        time.sleep(0.03)
```
- Ham kare 240×320×3 = **225 KB**, JPEG (kalite 95) ≈ **37 KB** → ~6x sıkıştırma.
- `global` olmazsa fonksiyon yerel bir değişken yaratır, paylaşılan kutu hep `None` kalır.
- `with lock:` olmazsa okuma/yazma çakışabilir (**race condition**).

### Parça 3 — Flask ve MJPEG canlı akış
```python
@app.route("/stream")
def stream():
    def generate():
        while True:
            with lock:
                jpg = latest_jpeg
            if jpg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000, threaded=True)
```
Terminal logu okuma: `"GET /stream HTTP/1.1" 200` → istek başarılı; `404` (favicon) → zararsız.

### Parça 4 — Etiket seçme ve çekim
- `/` (GET/POST): etiket formu + mevcut sınıflar ve sayıları
- `/capture`: canlı görüntü, sayaç, **Space** tuşu → `fetch('/snap')` (sayfa yenilenmez)
- `/snap` (POST): kutudaki son kareyi `dataset/<etiket>/<etiket>_YYYYMMDD-HHMMSS-mmm.jpg` olarak kaydeder

**Dokümana göre iyileştirmeler:**
| Doküman | Bu proje | Neden |
|---|---|---|
| Varsayılan 640×480 **kırpılmış** sensör modu | `sensor=(1640, 1232)` | Tam görüş alanı |
| Saniye hassasiyetinde dosya adı | **Milisaniyeli** ad | Aynı saniyede iki çekimde üzerine yazma olmasın |
| Çekimde kamerayı ikinci kez çağırma | Önizlemedeki **son kareyi** kaydetme | Ekranda görülen = kaydedilen; tek kamera erişimi |
| Her çekimde sayfa yenileme | `fetch` + Space | Hızlı, akıcı çekim |

Tam kod: [`get_img_data.py`](get_img_data.py)

---

## 4. P1.3 — Veri toplama

### 5 altın kural
| # | Kural | Neden |
|---|---|---|
| 1 | **Tutarlılık** — kamera sabit, aynı ayarlar | Canlı testte de aynı koşullar (*training/serving skew*) |
| 2 | **Çeşitlilik** — her 2–3 çekimde nesneyi döndür/kaydır | Tek pozu ezberlemesin |
| 3 | **Denge** — sınıf başına ~60 | Model çoğunluk sınıfına kaymasın |
| 4 | **Temizlik** — kadrajda yalnızca o sınıfın nesnesi, el yok | Yanlış etiket / *shortcut learning* |
| 5 | **Gerçekçi arka plan** — nesnelerin olduğu **aynı** yer | Model zemini değil nesneyi öğrensin |

**Odak:** Camera v2'nin odağı sabittir. Nesne kadrajın 1/3–1/2'sini kaplasın; hafif bulanıklık sorun değil (görüntü 160×160'a küçülecek). Tutarlılık, keskinlikten önemlidir.

### Sayım kontrolü
```bash
for d in dataset/*/; do echo "$d: $(ls "$d" | wc -l) görüntü"; done
# dataset/arka_plan/: 60 görüntü
# dataset/kamera/: 60 görüntü
# dataset/pi_kutusu/: 60 görüntü
```

---

## 5. P1.4 — Edge Impulse ile eğitim

### 5.1 Proje ve yükleme
- Proje: `raspi5-custom-classification` (Personal)
- **Data acquisition → Upload data**, **her sınıf ayrı ayrı**:
  - Upload mode: *Select a folder* → `dataset/<sınıf>`
  - Category: **Automatically split between training and testing** (%80 / %20)
  - Label: **Enter label** → sınıf adı (**boş bırakılırsa etiketsiz yüklenir!**)
- Sonuç: 180 öğe → eğitim 145 (48/47/50), test 35.

**Test kümesi neden en başta ayrılır?** Eğitimden sonra seçilirse (1) modelin gördüğü örnekler sınava girebilir, (2) test sonucuna göre ayar yapılırsa test artık tarafsız değildir (**veri sızıntısı**).

### 5.2 Impulse tasarımı
```
Image data (160×160, Squash) → Image (RGB) → Transfer Learning (Images) → 3 sınıf
```
| Ayar | Değer | Neden |
|---|---|---|
| Boyut | 160×160 | 224'e göre ~2x daha az hesap; nesneler büyük ve basit |
| Resize mode | **Squash** | Kırpma yok, hiçbir bölge kaybolmaz; eğitim ve çıkarımda **aynı** yöntem kullanılmalı |
| Color depth | **RGB** | Kırmızı kutu / yeşil kart → renk güçlü bir ipucu |

**Resize modları:**
| Mod | Ne yapar | Eksi |
|---|---|---|
| Squash | Kareye ezer | Nesneler hafif deforme |
| Fit short axis | Kısa kenara göre ölçekler, kırpar | Kenardaki nesneler kaybolabilir |
| Fit long axis | Uzun kenara göre ölçekler, siyahla doldurur | Boş şeritler |

**Hedef cihaz:** sağ üstte *Target* → **Raspberry Pi 5** (gecikme/bellek tahminleri hedefe göre yapılır).

### 5.3 Feature explorer yorumu
- 76 800 boyutlu veri (160×160×3) 2B'ye indirgenip çizilir; her nokta bir fotoğraf.
- Üç sınıf **ayrık kümeler** → yüksek doğruluk beklenir.
- Kümeler **çok sıkı** → aynı pozdaki fotoğraflar birbirine çok benziyor → test sonucu iyimser olabilir.

### 5.4 Eğitim ayarları
| Ayar | Değer | Anlamı |
|---|---|---|
| Model | MobileNetV2 160×160 1.0 | Backbone (ImageNet ağırlıkları), α = 1.0 |
| Training cycles | 20 | Epoch sayısı |
| Learning rate | 0.0005 | Küçük adım → önceden öğrenilmiş bilgiyi bozmamak |
| Validation | %20 | Eğitimi izlemek için ayrılan "deneme sınavı" |
| Data augmentation | ✅ | Rastgele flip, crop/zoom, parlaklık → çeşitlilik eksiğini kapatır |
| Final dense layer | 0 | Ek gizli katman yok |
| Dropout | 0.1 | Ezberlemeyi azaltır |

### 5.5 Eğitim sonucu
```
Epoch 10/10 - 4/4 - loss: 3.2885e-08 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
```
- `4/4` → epoch başına 4 batch (~116 görüntü / 32).
- Logda "Epoch x/10": Edge Impulse muhtemelen head eğitiminin ardından bir **fine-tuning** aşaması daha uyguluyor.
- `accuracy` ≈ `val_accuracy` → **overfitting belirtisi yok** (büyük fark olsaydı ezberleme olurdu).
- Validation confusion matrix: tamamen köşegen, F1 = 1.00.

---

## 6. P1.5 — Test

- **Model testing → Classify all:** 35 test görüntüsünde **%100**, belirsiz %0.
- `UNCERTAIN` sütunu: en yüksek olasılık güven eşiğinin altındaysa tahmin "belirsiz" sayılır.
- ⚠️ Gösterilen sonuç **float32** modele aittir; Pi'ye **int8** gidecek. (Arayüzde int8 seçeneği pasif olduğu için int8'i Pi'de kendimiz test ettik — *hedef donanımda ölç*.)

---

## 7. P1.6 — Modeli Pi'ye almak ve incelemek

### 7.1 İndirme
**Dashboard → Download block output:**
| Model | Boyut |
|---|---|
| TensorFlow Lite (float32) | 8.9 MB |
| TensorFlow Lite (int8 quantized) | 2.7 MB |

```bash
mkdir -p models
mv ~/Downloads/ei-...-float32-model.8.lite models/model_float32.tflite
mv ~/Downloads/ei-...-int8-quantized-model.8.lite models/model_int8.tflite
```

**Neden 4x değil 3.3x?** Bias'lar int32, kanal başına scale/zero_point (float32) ve model grafiği/metaverisi quantize edilmez.
**Neden MobileNet'in 3.5 MB'ından küçük?** 1001 sınıflık head (1280×1001 ≈ 1.28 M parametre) yerine 3 sınıflık head (1280×3 = 3 840).

### 7.2 Giriş/çıkış incelemesi
```
===== model_float32 =====
GİRİŞ : [1 160 160 3] float32 | (0.0, 0)
ÇIKIŞ : [1 3] float32         | (0.0, 0)

===== model_int8 =====
GİRİŞ : [1 160 160 3] int8 | (0.003921568859368563, -128)
ÇIKIŞ : [1 3] int8         | (0.00390625, -128)
```

**Giriş (int8):** scale = 1/255, zero_point = −128
- q = −128 → 0.0, q = 127 → 1.0 → Edge Impulse pikselleri **[0, 1]**'e ölçekler.
- Pratikte: **q = piksel − 128**.

**Çıkış (int8):** scale = 1/256, zero_point = −128 → [0, 0.996] → **zaten olasılık**; Edge Impulse softmax'ı modele eklemiş. **Softmax tekrar uygulanmaz.**

| Model | Ön işleme |
|---|---|
| float32 | `piksel / 255.0` |
| int8 | `piksel − 128` (önce int16'ya çevir — uint8 taşmasın!) |

### 7.3 float32 vs int8 — Pi 5'te ölçüm
[`compare_models.py`](compare_models.py), 180 görüntü, 4 thread:
```
model_float32  |  9.2 MB | çıkarım:  5.24 ± 0.68 ms | doğruluk: %100.0
model_int8     |  2.8 MB | çıkarım:  1.89 ± 0.16 ms | doğruluk: %100.0
İki modelin aynı cevabı verdiği görüntü oranı: %100.0
```
- int8: **2.8x hızlı, 3.3x küçük, doğruluk kaybı yok.**
- Ayrık kümeler sayesinde quantization'ın yuvarlama hataları hiçbir tahmini değiştirmedi.
- ⚠️ Görüntülerin çoğu eğitim verisi → doğruluk iyimser; bu testin değeri **hız ve uyum** ölçümünde.

---

## 8. P1.7 — Canlı sınıflandırma

[`live_classify.py`](live_classify.py): veri toplama aracıyla **aynı** kamera ayarları + her karede int8 çıkarım + sonucu görüntüye yazma.

```python
def classify(frame):
    x = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_AREA)   # squash
    x = (x.astype(np.int16) - 128).astype(np.int8)[None]
    itp.set_tensor(inp['index'], x); itp.invoke()
    return (itp.get_tensor(out['index'])[0].astype(np.float32) - z_out) * s_out
```
- **THRESHOLD = 0.80** → altında "belirsiz" (closed-set'e karşı önlem).
- Ekranda: etiket + güven + çıkarım süresi + FPS.
- OpenCV `putText` Türkçe karakter çizemez → `cikarim`.

**Sonuç:**
| Sahne | Tahmin | Çıkarım | Sistem FPS |
|---|---|---|---|
| Pi kutusu | `pi_kutusu` %99 | 3.6 ms | 15 |
| Kamera modülü | `kamera` %100 | 3.9 ms | 15 |
| Boş alan (köşede bir devre kartı görünürken) | `arka_plan` %95 | 3.4 ms | 15 |

**Yorum — pipeline darboğazı yine karşımızda:**
- Model tek başına ~270 FPS'e yetecek hızda, ama sistem **15 FPS**. Darboğaz model değil: kamera yakalama + `cvtColor` + JPEG encode + MJPEG akışı.
- Canlıda çıkarım 1.9 ms yerine ~3.5 ms: 4 çekirdek kamera, ISP sürücüsü, JPEG ve Flask thread'leriyle paylaşılıyor. İzole benchmark ≠ gerçek sistem.

```bash
python live_classify.py    # → http://<PI_IP>:5000
```

---

## 9. Yaşanan hatalar ve dersleri

| Hata | Belirti | Çözüm | Ders |
|---|---|---|---|
| Pi kutusu `kamera` etiketiyle çekildi | — | Veri seti silinip baştan toplandı | Yanlış etiket sessizce yanlış öğretir |
| `mv kamera pi_kutusu` (hedef varken) | Klasör içine taşınır | Önce `ls` ile kontrol | Komutların "sessiz" davranışlarını bil |
| `arka_plan` sandalyeden çekildi | (Model zemini öğrenirdi) | Aynı masadan yeniden çekildi | Negatif sınıf hedef ortamdan |
| Toplu yüklemede etiketler boş | Label sütunu boş | Silinip sınıf sınıf "Enter label" ile yüklendi | Gözetimli öğrenme etiket ister |
| `arka_paln` yazım hatası | Output features'ta görüldü | Edit labels ile düzeltildi | Erken yakalanan hata ucuz |
| `Device or resource busy` | Picamera2 açılamıyor | `pgrep -af python` → eski süreci kapat | Kamera tek sürece hizmet eder |

---

## 10. Terimler sözlüğü

| Terim | Açıklama |
|---|---|
| **Veri seti** | Etiketli örneklerin tamamı. |
| **Etiket** | Bir örneğin sınıf adı; burada klasör adı. |
| **Negatif / arka plan sınıfı** | "Hiçbiri yok" durumu. |
| **Supervised learning** | Etiketli (girdi → cevap) çiftlerinden öğrenme. |
| **Flask** | Python'da hafif web sunucusu kütüphanesi. |
| **HTTP, GET, POST** | Tarayıcı–sunucu iletişimi; GET "ver", POST "gönder/değiştir". |
| **Route** | Bir adresi bir Python fonksiyonuna bağlayan kural. |
| **Port / 0.0.0.0** | Programın dinlediği kapı numarası / tüm ağ arayüzlerinden bağlantı kabulü. |
| **MJPEG** | Ardışık JPEG'lerden oluşan basit video akışı (`multipart/x-mixed-replace`). |
| **Generator (yield)** | Değerleri istendikçe üreten fonksiyon. |
| **Thread / daemon thread** | Paralel iş parçacığı / ana programla birlikte biten thread. |
| **Lock / race condition** | Ortak veriye tek erişim kilidi / eşzamanlı erişimden doğan rastgele hata. |
| **QVGA** | 320×240 çözünürlük. |
| **Impulse** | Edge Impulse'ta ham veri → ön işleme → model → çıktı zinciri. |
| **Feature** | Modele giren her sayı (burada her pikselin her kanalı). |
| **Feature explorer** | Yüksek boyutlu özelliklerin 2B'ye indirgenmiş görselleştirmesi (UMAP/t-SNE). |
| **Squash** | Oranı korumadan kareye ezerek boyutlandırma. |
| **Epoch / batch** | Tüm verinin bir geçişi / bir güncellemedeki örnek grubu. |
| **Data augmentation** | Eğitimde rastgele dönüşümlerle yapay çeşitlilik. |
| **Fine-tuning** | Backbone'un son katmanlarını düşük öğrenme oranıyla yeniden eğitmek. |
| **Shortcut learning** | Modelin nesne yerine tesadüfi bir ipucunu (el, zemin) öğrenmesi. |
| **Veri sızıntısı** | Test bilgisinin eğitime/model seçimine karışması. |
| **Güven eşiği** | Altında tahminin "belirsiz" sayıldığı olasılık değeri. |

---

## 11. Mülakat notları

**"Transfer learning nedir, neden az veriyle çalışır?"**
ImageNet'te eğitilmiş backbone genel görsel özellikleri zaten çıkarabilir; yalnızca küçük bir head (burada 1280×3 + 3 parametre) eğitilir. Az parametre → az veriyle bile overfitting riski düşük.

**"%100 doğruluk aldım."**
İlk soru: Test verisi gerçekten bağımsız mı? Aynı oturumda, aynı açıdan çekilmiş neredeyse aynı kareler sonucu iyimser yapar. Gerçek genelleme için farklı gün/ışık/ortamdan test seti gerekir.

**"Yeni bir modeli entegre ederken ilk kontrol ettiğin 3 şey?"**
(1) Giriş şekli ve tipi, (2) normalizasyon ([0,1], [−1,1], mean/std), (3) çıkış logit mi olasılık mı. Bu projede MobileNet ile Edge Impulse modelinin üçü de farklıydı.

**"Training/serving skew nasıl önlenir?"**
Veri toplama ve çıkarımda aynı sensör, aynı kamera modu, aynı çözünürlük, aynı resize yöntemi ve aynı normalizasyon.

**"int8 quantization neden tam 4x küçültmez?"**
Bias'lar int32, kanal başına scale/zero_point float32, grafik ve metaveri quantize edilmez.

**"Shortcut learning örneği ver."**
Kurt/husky sınıflandırıcısının aslında karı öğrenmesi. Bu projede: arka plan farklı bir yerden çekilseydi model nesneye değil zemine bakardı.

**"Bir sınıflandırıcıyı sahada daha güvenilir nasıl yaparsın?"**
Güven eşiği + "belirsiz" çıktısı, ardışık karelerde oylama/yumuşatma, hedef ortamdan çeşitli veri, bağımsız test seti.
