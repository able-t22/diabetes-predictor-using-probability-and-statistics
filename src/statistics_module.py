import numpy as np
def calculate_statistics(data):
    stats = {}
    for col in data.columns[:-1]:
        stats[col] = {
            "mean": np.mean(data[col]),
            "std": np.std(data[col]),
            "variance": np.var(data[col]),
            "min": np.min(data[col]),
            "max": np.max(data[col])
        }
    return stats
