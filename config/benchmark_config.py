DEFAULT_CONFIG = {
    "models": ["google-bert/bert-base-uncased"],
    "batch_sizes": [1, 2, 4, 8],
    "sequence_lengths": [8, 32, 128, 512],
    "memory_threshold": 0.8  # 80% memory utilization threshold
}