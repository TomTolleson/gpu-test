import pytest
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
