import torch
import time
import mlx.core as mx
import numpy as np
from mlx_plot import plot_matmul_times
from tqdm import tqdm

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time)
        return result, execution_time
    return wrapper

@measure_time
def torch_cpu(a, b):
    return a @ b
@measure_time
def torch_mps(a, b):
    device = torch.device('mps')
    aa = a.to(device)
    bb = b.to(device)
    cc = aa @ bb
    torch.mps.synchronize() #Waits for all kernels in all streams on a MPS device to complete.
    return cc

@measure_time
def mlx_cpu(a, b):
    c = mx.matmul(a, b, stream=mx.cpu)
    mx.eval(c)
    return c

@measure_time
def mlx_gpu(a, b):
    c = mx.matmul(a, b, stream=mx.gpu)
    mx.eval(c)
    return c

@measure_time
def np_cpu(a,b):
    return a @ b

def err_norm(a, b):
    """
    Calculate the accuracy as the sum of absolute differences between two matrices.
    """
    if a.shape != b.shape:
        raise ValueError("Matrices must have the same shape to calculate accuracy.")
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays.")
    error = np.linalg.norm(a - b, ord='fro')
    return float(error)

def run_test(Ni:int, Nj:int):
    print(f"Matrix multiplication: {Ni}x{Nj} times {Nj}x{Ni}")
    n_run = 10
    numpy_times = []
    torch_acc, torch_times = [], []
    torch_mps_acc, torch_mps_times = [], []
    mlx_acc, mlx_times = [], []
    mlx_gpu_acc, mlx_gpu_times = [], []
    # Run the tests
    for n in tqdm(range(n_run), desc="Running tests", unit="run"):
        A = np.random.rand(Ni, Nj).astype(np.float32)
        B = np.random.rand(Nj, Ni).astype(np.float32)
        torch_A = torch.from_numpy(A)
        torch_B = torch.from_numpy(B)
        A_mlx = mx.array(A, dtype=mx.float32)
        B_mlx = mx.array(B, dtype=mx.float32)
        
        C_numpy, numpy_time = np_cpu(A, B)
        
        C_torch, torch_time = torch_cpu(torch_A, torch_B)
        torch.mps.synchronize() #Waits for all kernels in all streams on a MPS device to complete.
        C_torch_mps, torch_mps_time = torch_mps(torch_A, torch_B)
        # Check if C and C_mps are close
        if torch.allclose(C_torch, C_torch_mps.cpu(), atol=1e-6):
            pass
        else:
            print("Torch calculation out of tolerance.")
            
        C_mlx, mlx_time = mlx_cpu(A_mlx, B_mlx)
        C_mlx_gpu, mlx_gpu_time = mlx_gpu(A_mlx, B_mlx)
        if mx.allclose(C_mlx, C_mlx_gpu):
            pass
        else:
            print("MLX calculation out of tolerance.")
            
        numpy_times.append(numpy_time)
        torch_times.append(torch_time)
        torch_mps_times.append(torch_mps_time)
        mlx_times.append(mlx_time)
        mlx_gpu_times.append(mlx_gpu_time)
        
        torch_acc.append(err_norm(C_numpy, C_torch.numpy()))
        torch_mps_acc.append(err_norm(C_numpy, C_torch_mps.cpu().numpy()))
        mlx_acc.append(err_norm(C_numpy, np.array(C_mlx)))
        mlx_gpu_acc.append(err_norm(C_numpy, np.array(C_mlx_gpu)))
        
    accuracy = [torch_acc,
                torch_mps_acc,
                mlx_acc,
                mlx_gpu_acc]
    runtimes = [numpy_times, 
                torch_times, 
                torch_mps_times, 
                mlx_times, 
                mlx_gpu_times]
    return accuracy, runtimes
    

if __name__ == "__main__":
    print("MLX Tutorial")
    print("=============")
    print("This script runs matrix multiplication tests on CPU and GPU using PyTorch and MLX.")
    print("It measures the execution time for each operation.")
    print("The results are saved in mlx_tutorial.json.")

    matrix_sizes = [(512, 8192),
                    (8192, 512), 
                    (1024, 1024), 
                    (2048, 2048), 
                    (4096, 4096),
                    (8192, 8192),
                    (8192*2, 8192*2)]
    
    accuracy = {}
    times = {}
    for Ni, Nj in matrix_sizes:
        print(f"Running test for matrix size: {Ni}x{Nj}")
        acc, runtimes = run_test(Ni, Nj)
        accuracy.update({f"{Ni}x{Nj}" : acc})
        times.update({f"{Ni}x{Nj}" : runtimes})
    import json
    with open('mlx_times.json', 'w') as f:
        json.dump(times, f, indent=4)
    with open('mlx_accuracy.json', 'w') as f:
        json.dump(accuracy, f, indent=4)
        
    #plot_matmul_times(times)