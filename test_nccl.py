import os
import torch
import torch.distributed as dist

def run():
    print("1. Starting NCCL Test...")
    # Force settings to prevent hang
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # Initialize Process Group
    print("2. Initializing Process Group (If it hangs here, it's a Network/Driver issue)...")
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23456", world_size=2, rank=0)
    
    print("3. Doing a dummy tensor computation...")
    # Put something on GPU 0 and GPU 1
    t1 = torch.ones(1).to(0)
    t2 = torch.ones(1).to(1)
    
    print("4. Attempting All-Reduce...")
    # This triggers the actual GPU-to-GPU communication
    dist.all_reduce(t1)
    dist.all_reduce(t2)
    
    print("5. SUCCESS! NCCL is working.")
    dist.destroy_process_group()

if __name__ == "__main__":
    # Simulate 2 processes on 1 machine
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    # We spawn a subprocess for rank 1, but for a quick test, 
    # we just try to init rank 0 to see if it binds the driver.
    # A full test requires mp.spawn, but let's see if init hangs first.
    try:
        run()
    except Exception as e:
        print(f"FAILED: {e}")