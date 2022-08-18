import numpy as np

input_path = '/home/fengwen/compare_bn/oneflow-input.txt'
running_mean_path = '/home/fengwen/compare_bn/oneflow-running_mean.txt'
running_var_path = '/home/fengwen/compare_bn/oneflow-running_var.txt'
# (16, 32, 320, 320)
# (32,)
# (32,)

def torch_cal(input):
    import torch
    input = torch.FloatTensor(input).reshape(16, 32, 320, 320).cuda()
    bn = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    running_mean = np.loadtxt(running_mean_path).reshape(32,)
    running_var = np.loadtxt(running_var_path).reshape(32,) # .resize((1, 32, 128, 128))

    bn.running_mean = torch.FloatTensor(running_mean).cuda()
    bn.running_var = torch.FloatTensor(running_var).cuda()

    return bn(input).cpu().detach().numpy()

def oneflow_cal(input):
    import oneflow as torch
    input = torch.FloatTensor(input).reshape(16, 32, 320, 320).cuda()
    bn = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    running_mean = np.loadtxt(running_mean_path).reshape(32,)
    running_var = np.loadtxt(running_var_path).reshape(32,) # .resize((1, 32, 128, 128))

    bn.running_mean = torch.FloatTensor(running_mean).cuda()
    bn.running_var = torch.FloatTensor(running_var).cuda()

    return bn(input).cpu().detach().numpy()

# input = np.loadtxt(input_path)#.resize((1, 32, 128, 128)) # (1, 32, 128, 128)
# inputflie = '/home/fengwen/compare_bn/out'
# np.save(inputflie, input)
# # np.save(outputfile,input)
# # with open(inputflie,'rb') as f:
# #     input = np.load(f)
# # print("load is ok")
input_path = '/home/fengwen/compare_bn/out.npy'
with open(input_path, 'rb') as f:
    input = np.load(f)

print('load is ok')
a = torch_cal(input).flatten()
# b = oneflow_cal(input).flatten()
b = a

print(len(a), len(b))
res = np.allclose(a, b)
print(res)
print(np.mean(np.abs(a-b)))
# (1, 32, 128, 128)
# (32,)
# (32,)

