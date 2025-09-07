import torch
from thop import profile
from modules.GMSNet import GMSNet

GMSNet = GMSNet()

checkpoint = torch.load('weights/outdoor.pth')
GMSNet.net.load_state_dict(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GMSNet.net.to(device)
GMSNet.net.eval()

x = torch.randn(1, 3, 480, 640).to(device)

flops, params = profile(GMSNet.net, inputs=(x,))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")
