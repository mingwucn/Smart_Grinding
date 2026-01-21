import os
import dill as pickle
from tqdm import tqdm
import lzma

def save_chunk(data_list, directory, base_filename, chunk_idx):
    """Save a chunk of data to a file."""
    filepath = os.path.join(directory, f"{base_filename}_chunk{chunk_idx}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(data_list, f)
    print(f"Saved chunk {chunk_idx} to {filepath}")

def save_compressed_chunk(data_list, directory, base_filename, chunk_idx):
    """Save a compressed chunk of data to a file."""
    filepath = os.path.join(directory, f"{base_filename}_chunk{chunk_idx}.xz")
    with lzma.open(filepath, "wb") as f:
        pickle.dump(data_list, f)
    print(f"Saved compressed chunk {chunk_idx} to {filepath}")

def save_compressed_chunk_subset(data_list, directory, base_filename, subset_idx, chunk_idx):
    """Save a compressed chunk of data to a file with subset-specific naming."""
    filepath = os.path.join(directory, f"{base_filename}_subset_{subset_idx}_chunk{chunk_idx}.xz")
    with lzma.open(filepath, "wb") as f:
        pickle.dump(data_list, f)
    print(f"Saved subset {subset_idx} chunk {chunk_idx} to {filepath}")

def load_chunks(directory, base_filename):
    chunk_idx = 0
    data = []
    while True:
        filepath = os.path.join(directory, f"{base_filename}_chunk{chunk_idx}.pkl")
        if not os.path.exists(filepath):
            break
        with open(filepath, "rb") as f:
            data.extend(pickle.load(f))
        chunk_idx += 1
    return data

def load_compressed_chunks(directory, base_filename):
    chunk_idx = 0
    data = []
    while True:
        filepath = os.path.join(directory, f"{base_filename}_chunk{chunk_idx}.xz")
        if not os.path.exists(filepath):
            break
        with lzma.open(filepath, "rb") as f:
            data.extend(pickle.load(f))
        chunk_idx += 1
    return data

def load_compressed_chunks_subset(directory, base_filename, subset_idx):
    """Load all compressed chunks for a specific subset."""
    subset_data = []
    chunk_idx = 0

    while True:
        filepath = os.path.join(directory, f"{base_filename}_subset_{subset_idx}_chunk{chunk_idx}.xz")
        if not os.path.exists(filepath):
            break  # Exit loop if no more chunks are found
        with lzma.open(filepath, "rb") as f:
            chunk_data = pickle.load(f)
            subset_data.extend(chunk_data)
        chunk_idx += 1

    return subset_data

def load_unified_dataset(directory, base_filename, num_subsets):
    """Load and unify all subsets into a single dataset."""
    unified_data = []

    for subset_idx in range(num_subsets):
        subset_data = load_compressed_chunks(directory, base_filename, subset_idx)
        unified_data.extend(subset_data)

    return unified_data
