import torchvision
import torch
from time import perf_counter
import numpy as np

def timer(f,*args):
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))

# Example 1.1 Pytorch cpu version
model_ft = torchvision.models.resnet18(pretrained=True)
model_ft.eval()
x_ft = torch.rand(1, 3, 224, 224)
eager_cpu_runtimes = [timer(model_ft, x_ft) for _ in range(105)]
print("Eager mode + CPU running time:", np.mean(eager_cpu_runtimes[5::]))

# Example 1.2 Pytorch gpu version
model_ft_gpu = torchvision.models.resnet18(pretrained=True).cuda()
x_ft_gpu = x_ft.cuda()
model_ft_gpu.eval()
eager_gpu_runtimes = [timer(model_ft_gpu, x_ft_gpu) for _ in range(105)]
print("Eager mode + GPU running time:", np.mean(eager_gpu_runtimes[5::]))


# Example 2.1 torch.jit.script cpu version
script_cell = torch.jit.script(model_ft)
script_cpu_runtimes = [timer(script_cell, x_ft) for _ in range(105)]
print("torch.jit.script() + CPU running time:", np.mean(script_cpu_runtimes[5::]))

# Example 2.2 torch.jit.script gpu version
script_cell_gpu = torch.jit.script(model_ft_gpu, x_ft_gpu)
script_gpu_runtimes = [timer(script_cell_gpu, x_ft_gpu) for _ in range(105)]
print("torch.jit.script() + GPU running time:", np.mean(script_gpu_runtimes[5::]))

import pdb; pdb.set_trace()