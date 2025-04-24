import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
        
def plot_matmul_times(data):
    # Extract matrix sizes and execution times
    matrix_sizes = list(data.keys())
    numpy_times = [times[0] for times in data.values()]
    torch_cpu_times = [times[1] for times in data.values()]
    torch_mps_times = [times[2] for times in data.values()]
    mlx_cpu_times = [times[3] for times in data.values()]
    mlx_gpu_times = [times[4] for times in data.values()]

    # Plot the data
    sns.set_theme(style="ticks")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(matrix_sizes))
    shift = 0.05
    plot_with_bars = lambda x, y, label, marker: plt.errorbar(x=x, y=np.mean(y, axis=1), yerr=np.std(y, axis=1), label=label, marker=marker, capsize=5)
    plot_with_bars(x, numpy_times, label='NumPy', marker='x')
    plot_with_bars(x+shift, torch_cpu_times, label='Torch CPU', marker='o')
    plot_with_bars(x+2*shift, torch_mps_times, label='Torch MPS', marker='o')
    plot_with_bars(x+3*shift, mlx_cpu_times, label='MLX CPU', marker='+')
    plot_with_bars(x+4*shift, mlx_gpu_times, label='MLX GPU', marker='+')

    # Add labels, title, and legend
    plt.xticks(x, matrix_sizes, rotation=45)
    plt.xlabel('Matrix Sizes')
    plt.ylabel('Execution Time (seconds)')
    plt.yscale('log')  # Use logarithmic scale for better visibility
    plt.title('Matrix Multiplication Execution Times')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.savefig('mlx_times.png')
    plt.show()

def plot_speedup(data):
    # Extract matrix sizes and execution times
    matrix_sizes = list(data.keys())
    numpy_times = np.mean([times[0] for times in data.values()], axis=1)
    torch_cpu_times = np.mean([times[1] for times in data.values()], axis=1)
    torch_mps_times = np.mean([times[2] for times in data.values()], axis=1)
    mlx_cpu_times = np.mean([times[3] for times in data.values()], axis=1)
    mlx_gpu_times = np.mean([times[4] for times in data.values()], axis=1)

    # Calculate speedup
    speedup_torch_cpu = numpy_times / torch_cpu_times
    speedup_torch_mps = numpy_times / torch_mps_times
    speedup_mlx_cpu = numpy_times / mlx_cpu_times
    speedup_mlx_gpu = numpy_times / mlx_gpu_times

    # Plot the speedup
    sns.set_theme(style="ticks")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(matrix_sizes))
    plt.bar(x - 0.2, speedup_torch_cpu, width=0.2, label='Torch CPU', color='blue')
    plt.bar(x, speedup_torch_mps, width=0.2, label='Torch MPS', color='orange')
    plt.bar(x + 0.2, speedup_mlx_cpu, width=0.2, label='MLX CPU', color='green')
    plt.bar(x + 0.4, speedup_mlx_gpu, width=0.2, label='MLX GPU', color='red')
    # Add baseline at y=1.0
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='NumPy (1.0x)')
    # Add labels and title
    plt.xticks(x, matrix_sizes, rotation=45)
    plt.xlabel('Matrix Sizes')
    plt.ylabel('Speedup factor (x)')
    plt.title('Average Speedup to NumPy')
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('mlx_speedup.png')
    plt.show()
    
if __name__ == "__main__":
    # Load the data from the JSON file
    with open('mlx_times.json', 'r') as f:
        data = json.load(f)
    # Plot the matrix multiplication times
    plot_matmul_times(data)
    plot_speedup(data)