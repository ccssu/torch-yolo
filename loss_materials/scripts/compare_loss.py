import matplotlib.pyplot as plt

of_losses_path = '/home/fengwen/one-yolo/runs/train/exp/results.csv'
torch_losses_path = '/home/fengwen/torch-yolo5/runs/train/exp/results.csv'

of_losses = []
torch_losses = []
names = [' ','box','obj','cls']
pos = 3
with open(of_losses_path, "r") as lines:
    next(lines)
    for line in lines:
        strs = line.split(',')
        line = strs[pos] 
        line = line.strip()
        of_losses.append(float(line))

with open(torch_losses_path, "r") as lines:
    next(lines)
    for line in lines:
        strs = line.split(',')
        line = strs[pos] 
        line = line.strip()
        torch_losses.append(float(line))



r = 298
indes = [i for i in range(r)]

of_losses = of_losses[0:r].copy()
torch_losses = torch_losses[0:r].copy()

plt.plot(indes, of_losses, label="oneflow")
plt.plot(indes, torch_losses, label="pytorch")

plt.xlabel("iter - axis")
# Set the y axis label of the current axis.
plt.ylabel("loss - axis")
# Set a title of the current axes.
plt.title("compare ")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
plt.savefig('/home/fengwen/loss_materials/imgs/'+names[pos]+'.jpg')