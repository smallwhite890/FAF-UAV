import torch
from thop  import profile
from models.experimental import attempt_load
mode_path = "/home/raytrack/桌面/SEyolov7-main/runs/train/exp7/weights/best.pt"
model = attempt_load(mode_path,map_location=torch.device('cpu'))
input = torch.randn(12,3,640,640)
flops,params = profile(model,inputs=(input, ),verbose=False)
gflops = flops/1e9
print(f"Total GFLOPs: {gflops:.3f} Flops")
print(f"Total Params: {params} params")
