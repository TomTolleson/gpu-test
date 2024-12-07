import os
import logging

def create_test_file():
    tests_dir = 'tests'
    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)

    test_file_path = os.path.join(tests_dir, 'test_gpu_benchmark.py')
    with open(test_file_path, 'w') as f:
        f.write("""import pytest
import torch
from src.gpu_benchmark import run_benchmark, check_gpu_availability

class TestGPUBenchmarks:
    def test_gpu_availability(self):
        assert check_gpu_availability() is not None

    def test_run_benchmark(self):
        result = run_benchmark()
        assert result is not None

    def test_memory_usage(self):
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            result = run_benchmark()
            memory_after = torch.cuda.memory_allocated()
            assert memory_after >= memory_before
""")

def create_gpu_benchmark_file():
    src_dir = 'src'
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)

    benchmark_file_path = os.path.join(src_dir, 'gpu_benchmark.py')
    with open(benchmark_file_path, 'w') as f:
        f.write("""import torch
import logging
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

logger = logging.getLogger(__name__)

def check_gpu_availability():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            
            logger.info(f"GPU Available: {device_name}")
            logger.info(f"Number of devices: {device_count}")
            logger.info(f"Memory allocated: {memory_allocated / 1024**2:.2f} MB")
            logger.info(f"Memory reserved: {memory_reserved / 1024**2:.2f} MB")
            
            return {
                'device_name': device_name,
                'device_count': device_count,
                'memory_allocated': memory_allocated,
                'memory_reserved': memory_reserved
            }
        else:
            logger.error("No GPU available")
            return None
    except Exception as e:
        logger.error(f"Error checking GPU availability: {str(e)}")
        return None

def run_benchmark(model_name="google-bert/bert-base-uncased", batch_sizes=[1, 2, 4]):
    try:
        gpu_status = check_gpu_availability()
        if not gpu_status:
            raise RuntimeError("GPU is required for benchmarking")

        args = PyTorchBenchmarkArguments(
            models=[model_name],
            batch_sizes=batch_sizes,
            sequence_lengths=[8, 32, 128, 512],
            training=True,
            inference=True,
            memory=True,
            multi_process=False,
        )
        
        logger.info(f"Starting benchmark for model: {model_name}")
        benchmark = PyTorchBenchmark(args)
        result = benchmark.run()
        
        # Save results
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results/benchmark_result_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"Benchmark results saved to {result_file}")
        return result

    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        raise
""")

def setup_logging():
    logging_config = """import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/gpu_benchmark_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info('Logging initialized')
    return logger
"""
    
    with open(os.path.join('src', 'logging_config.py'), 'w') as f:
        f.write(logging_config)

def create_results_directory():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create .gitkeep to track empty directory
    with open(os.path.join(results_dir, '.gitkeep'), 'w') as f:
        pass

def create_requirements_file():
    requirements = """torch>=2.0.0
transformers>=4.33.2
pytest>=7.2.0
numpy>=1.24.0
tqdm>=4.65.0
datasets>=2.14.0
accelerate>=0.21.0
bitsandbytes>=0.41.1
optimum-nvidia>=0.1.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)

def main():
    # Create all necessary directories and files
    create_test_file()
    create_gpu_benchmark_file()
    setup_logging()
    create_results_directory()
    create_requirements_file()
    
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    print("Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Create and activate a virtual environment")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Run tests: pytest tests/")
    print("4. Run benchmark: python -c 'from src.gpu_benchmark import run_benchmark; run_benchmark()'")

if __name__ == "__main__":
    main()
