import torch
import time

def check_gpu_availability():
    """Check if GPU is available and return device info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        return {
            "device": device,
            "name": device_name,
            "cuda_available": True
        }
    return {
        "device": torch.device("cpu"),
        "name": "CPU",
        "cuda_available": False
    }

def run_benchmark(matrix_size=1000, iterations=10):
    """Run a simple matrix multiplication benchmark."""
    device_info = check_gpu_availability()
    device = device_info["device"]
    
    # Create random matrices
    matrix_a = torch.rand(matrix_size, matrix_size, device=device)
    matrix_b = torch.rand(matrix_size, matrix_size, device=device)
    
    # Warmup
    torch.matmul(matrix_a, matrix_b)
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        result = torch.matmul(matrix_a, matrix_b)
        torch.cuda.synchronize() if device.type == "cuda" else None
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    return {
        "device": device_info["name"],
        "matrix_size": matrix_size,
        "iterations": iterations,
        "average_time": avg_time,
        "cuda_available": device_info["cuda_available"]
    }

if __name__ == "__main__":
    # Print GPU information
    gpu_info = check_gpu_availability()
    print(f"Device: {gpu_info['name']}")
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    
    # Run benchmark
    results = run_benchmark()
    print("\nBenchmark Results:")
    print(f"Device: {results['device']}")
    print(f"Matrix Size: {results['matrix_size']}x{results['matrix_size']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Average Time: {results['average_time']:.4f} seconds")