import numpy as np


# oneflow = np.loadtxt('/home/fengwen/loss_materials/temp/imgs-onflow.txt')
# torch = np.loadtxt('/home/fengwen/loss_materials/temp/imgs-torch.txt')

# print(np.allclose(oneflow, torch))


ROOT = '/home/fengwen/np_list'

for i in range(24):
    a = np.loadtxt(ROOT + '/flow' + str(i) + '.txt')
    b = np.loadtxt(ROOT + '/torch' + str(i) + '.txt')
    # print(i, np.allclose(a, b,0.00001,0.00001))
    # print(i, np.allclose(a, b,0.0000001,0.0000001) )
    print(i, np.allclose(a,  b))