import numpy as np
import torch

torch_imgs = torch.FloatTensor(np.ones([16,3,640,640])).cuda()
torch_m = torch.nn.Conv2d(3, 32, 6, 6, 2, 2, bias=False)
torch_weight_np = np.loadtxt('/home/fengwen/compare_model/torch-yolo.txt').reshape(32, 3, 6, 6)
torch_m.weight = torch.nn.Parameter(torch.tensor(torch_weight_np, dtype=torch.float32))
torch_m.to("cuda")
torch_silu = torch.nn.SiLU()
torch_y = torch_silu(torch_m(torch_imgs))
res1 = torch_y.cpu().detach().numpy().flatten()
print('pytorch results: ', torch_y.cpu().detach().flatten()[:50])



import oneflow as torch
torch_imgs = torch.FloatTensor(np.ones([16,3,640,640])).cuda()
torch_m = torch.nn.Conv2d(3, 32, 6, 6, 2, 2, bias=False)
torch_weight_np = np.loadtxt('/home/fengwen/compare_model/one-yolo.txt').reshape(32, 3, 6, 6)
torch_m.weight = torch.nn.Parameter(torch.tensor(torch_weight_np, dtype=torch.float32))
torch_m.to("cuda")
torch_silu = torch.nn.SiLU()
torch_y = torch_silu(torch_m(torch_imgs))
res2 = torch_y.cpu().detach().numpy().flatten()
print('oneflow results', torch_y.cpu().detach().flatten()[:50])

print(np.allclose(res1, res2))