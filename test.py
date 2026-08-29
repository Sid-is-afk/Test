import torch
from ultralytics import YOLO

# 1. Verify your RTX 3050 GPU is detected by Python
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    device_id = 0
else:
    print("Warning: GPU not detected. Defaulting to CPU.")
    device_id = "cpu"

# 2. Load your 39MB best.pt model
model = YOLO("best (1).pt")

# 3. Export to ONNX using your GPU with FP16 precision enabled
# 'half=True' ensures it stays light (~39MB) instead of doubling to 79MB
# 'simplify=True' removes redundant neural network layers for faster deployment
print("Starting GPU-accelerated ONNX export...")
onnx_path = model.export(
    format="onnx", 
    half=True, 
    simplify=True, 
    device=device_id
)

print(f"\nSuccess! Your light, fast deployment file is ready at: {onnx_path}")
