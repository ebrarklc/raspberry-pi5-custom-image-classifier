"""inspect_models.py: .tflite modellerinin giriş/çıkış tipini ve quantization parametrelerini yazdırır."""
from ai_edge_litert.interpreter import Interpreter

for name in ["model_float32", "model_int8"]:
    itp = Interpreter(model_path=f"models/{name}.tflite")
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    print(f"\n===== {name} =====")
    print("GİRİŞ :", inp['shape'], inp['dtype'].__name__, "| quantization (scale, zero_point):", inp['quantization'])
    print("ÇIKIŞ :", out['shape'], out['dtype'].__name__, "| quantization (scale, zero_point):", out['quantization'])
