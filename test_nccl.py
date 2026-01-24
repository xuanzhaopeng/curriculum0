import os
import torch
import torch.distributed as dist
import multiprocessing as mp

def run_check(rank, world_size):
    # 强制禁用 P2P 和 IB，只测最基础的 TCP 通信
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    print(f"Rank {rank}: Initializing...")
    try:
        # 设定 10 秒超时，避免无限死等
        dist.init_process_group("nccl", rank=rank, world_size=world_size, 
                                init_method="env://", timeout=torch.distributed.DEFAULT_PG_TIMEOUT)
        print(f"Rank {rank}: Success!")
    except Exception as e:
        print(f"Rank {rank}: Failed - {e}")

if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        print("Error: Need at least 2 GPUs to test NCCL.")
    else:
        print("Starting 2-GPU NCCL Check...")
        mp.spawn(run_check, args=(2,), nprocs=2, join=True)