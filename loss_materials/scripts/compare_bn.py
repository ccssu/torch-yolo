import numpy as np
import time

in_channels = 3
out_channels = 32
batch_size = 16

input_arr =  [16, 3, 640, 640]
# weight_arr = [32, 3, 6, 6]
def torch_cal(np_arr: np.number, weight_up = None)-> np.number:
    import torch
    torch_imgs = torch.FloatTensor(np_arr).cuda()
    # print(torch_imgs.dtype)
    # torch_m = torch.nn.Conv2d(in_channels, out_channels, 6, 6, 2, 2, bias=False)
    # torch_m.weight = torch.nn.Parameter(torch.tensor(weight_up, dtype=torch.float32))
    # torch_m.to("cuda")
    torch_bn = torch.nn.BatchNorm2d(3).cuda()
    torch_y = torch_bn(torch_imgs)
    return torch_y.cpu().detach().numpy()

def oneflow_cal(np_arr: np.number, weight_up = None)-> np.number:
    import oneflow as torch
    torch_imgs = torch.FloatTensor(np_arr).cuda()
    # print(torch_imgs.dtype)
    # torch_m = torch.nn.Conv2d(in_channels, out_channels, 6, 6, 2, 2, bias=False)
    # torch_m.weight = torch.nn.Parameter(torch.tensor(weight_up, dtype=torch.float32))
    # torch_m.to("cuda")
    torch_bn = torch.nn.BatchNorm2d(3).cuda()
    torch_y = torch_bn(torch_imgs)
    return torch_y.cpu().detach().numpy()


def test_01(len : int):
    for i in range(10):
        np_arr = np.random.random(input_arr)
        a = torch_cal(np_arr)
        b = oneflow_cal(np_arr)
        res = np.allclose(a, b)
        print(res)
        if res == False:
            print(a.flatten().tolist()[0:15])
            print(b.flatten().tolist()[0:15])
            break
           



# 随机生成 20 numpy测试 
test_01(20)

# import numpy as np
# import torch

# torch_imgs = torch.FloatTensor(np.ones([16,3,640,640])).cuda()
# torch_m = torch.nn.Conv2d(3, 32, 6, 6, 2, 2, bias=False)
# torch_weight_np = np.loadtxt('/home/fengwen/compare_model/torch-yolo.txt').reshape(32, 3, 6, 6)
# torch_m.weight = torch.nn.Parameter(torch.tensor(torch_weight_np, dtype=torch.float32))
# torch_m.to("cuda")
# torch_silu = torch.nn.SiLU()
# torch_y = torch_silu(torch_m(torch_imgs))
# res1 = torch_y.cpu().detach().numpy().flatten()
# print('pytorch results: ', torch_y.cpu().detach().flatten()[:50])



# import oneflow as torch
# torch_imgs = torch.FloatTensor(np.ones([16,3,640,640])).cuda()
# torch_m = torch.nn.Conv2d(3, 32, 6, 6, 2, 2, bias=False)
# torch_weight_np = np.loadtxt('/home/fengwen/compare_model/torch-yolo.txt').reshape(32, 3, 6, 6)
# torch_m.weight = torch.nn.Parameter(torch.tensor(torch_weight_np, dtype=torch.float32))
# torch_m.to("cuda")
# torch_silu = torch.nn.SiLU()
# torch_y = torch_silu(torch_m(torch_imgs))
# res2 = torch_y.cpu().detach().numpy().flatten()
# print('oneflow results', torch_y.cpu().detach().flatten()[:50])

# # print(np.allclose(res1, res2))
# import numpy as np
# import torch

# torch_imgs = torch.FloatTensor(np.ones([16,3,640,640])).cuda()
# torch_m = torch.nn.Conv2d(3, 32, 6, 6, 2, 2, bias=False)
# torch_weight_np = np.loadtxt('/home/fengwen/compare_model/torch-yolo.txt').reshape(32, 3, 6, 6)
# torch_m.weight = torch.nn.Parameter(torch.tensor(torch_weight_np, dtype=torch.float32))
# torch_m.to("cuda")
# torch_silu = torch.nn.SiLU()
# torch_y = torch_silu(torch_m(torch_imgs))
# res1 = torch_y.cpu().detach().numpy().flatten()
# print('pytorch results: ', torch_y.cpu().detach().flatten()[:50])



# import oneflow as torch
# torch_imgs = torch.FloatTensor(np.ones([16,3,640,640])).cuda()
# torch_m = torch.nn.Conv2d(3, 32, 6, 6, 2, 2, bias=False)
# torch_weight_np = np.loadtxt('/home/fengwen/compare_model/torch-yolo.txt').reshape(32, 3, 6, 6)
# torch_m.weight = torch.nn.Parameter(torch.tensor(torch_weight_np, dtype=torch.float32))
# torch_m.to("cuda")
# torch_silu = torch.nn.SiLU()
# torch_y = torch_silu(torch_m(torch_imgs))
# res2 = torch_y.cpu().detach().numpy().flatten()
# print('oneflow results', torch_y.cpu().detach().flatten()[:50])

# print(np.allclose(res1, res2))