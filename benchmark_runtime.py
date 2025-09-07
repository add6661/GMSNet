import torch
import time
from modules.GMSNet import GMSNet

class GMSNetForBenchmark(GMSNet):
    def forward(self, x):
        return self.net(x)

model = GMSNetForBenchmark()
model.eval()


state_dict = torch.load("weights/outdoor.pth", map_location='cpu')
model.net.load_state_dict(state_dict)
print("✔")

input_tensor = torch.randn(1, 1, 480, 640)

model_cpu = model.to('cpu')
input_cpu = input_tensor.to('cpu')

with torch.no_grad():
    for _ in range(10):
        _ = model_cpu(input_cpu)

    start = time.perf_counter()
    for _ in range(100):
        _ = model_cpu(input_cpu)
    end = time.perf_counter()

    cpu_time_ms = (end - start) / 100 * 1000
    print(f"[CPU]{cpu_time_ms:.2f} ms")

if torch.cuda.is_available():
    model_gpu = model.to('cuda')
    input_gpu = input_tensor.to('cuda')

    with torch.no_grad():
        for _ in range(10):
            _ = model_gpu(input_gpu)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = model_gpu(input_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()

        gpu_time_ms = (end - start) / 100 * 1000
        print(f"[GPU]{gpu_time_ms:.2f} ms")
else:
    print("⚠️ error")
