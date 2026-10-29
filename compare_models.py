"""compare_models.py: float32 ve int8 modellerini boyut, hız, doğruluk ve uyum açısından karşılaştırır."""
import os, glob, time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

LABELS = ["arka_plan", "kamera", "pi_kutusu"]   # Edge Impulse'daki ALFABETİK sıra!


def load(path):
    itp = Interpreter(model_path=path, num_threads=4)
    itp.allocate_tensors()
    return itp, itp.get_input_details()[0], itp.get_output_details()[0]


def preprocess(img, inp):
    """Edge Impulse ile AYNI ön işleme: 160x160 squash, RGB"""
    x = np.array(img.convert("RGB").resize((160, 160)))     # resize = squash (oran korunmaz)
    if inp['dtype'] == np.int8:
        x = (x.astype(np.int16) - 128).astype(np.int8)       # int8: piksel - 128
    else:
        x = x.astype(np.float32) / 255.0                      # float32: piksel / 255
    return np.expand_dims(x, 0)


def predict(itp, inp, out, x):
    itp.set_tensor(inp['index'], x)
    itp.invoke()
    y = itp.get_tensor(out['index'])[0]
    if out['dtype'] == np.int8:                              # dequantize
        s, z = out['quantization']
        y = (y.astype(np.float32) - z) * s
    return y                                                 # softmax YOK: zaten olasılık


# Veri setindeki tüm görüntüler (etiket = klasör adı)
files = sorted(glob.glob("dataset/*/*.jpg"))
truth = [os.path.basename(os.path.dirname(f)) for f in files]
images = [Image.open(f) for f in files]
print(f"{len(files)} görüntü yüklendi\n")

results = {}
for name in ["model_float32", "model_int8"]:
    itp, inp, out = load(f"models/{name}.tflite")
    xs = [preprocess(im, inp) for im in images]
    predict(itp, inp, out, xs[0])                            # warm-up
    preds, times = [], []
    for x in xs:
        t0 = time.perf_counter()
        p = predict(itp, inp, out, x)
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(LABELS[int(np.argmax(p))])
    acc = np.mean([p == t for p, t in zip(preds, truth)]) * 100
    results[name] = preds
    size = os.path.getsize(f"models/{name}.tflite") / 1e6
    print(f"{name:14s} | {size:4.1f} MB | çıkarım: {np.mean(times):5.2f} ± {np.std(times):.2f} ms | doğruluk: %{acc:.1f}")

agree = np.mean([a == b for a, b in zip(results["model_float32"], results["model_int8"])]) * 100
print(f"\nİki modelin aynı cevabı verdiği görüntü oranı: %{agree:.1f}")
