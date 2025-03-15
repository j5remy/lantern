import cupy as cp

def check_gpu():
    try:
        # Get number of available GPUs
        n_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"Number of GPUs available: {n_gpus}")
        
        # Get information about each GPU
        for i in range(n_gpus):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nGPU {i}: {props['name'].decode()}")
            print(f"Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
            print(f"Compute Capability: {props['major']}.{props['minor']}")
            
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        print("No CUDA-compatible GPU found")
        return False