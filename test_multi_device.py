import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    
    # 强制设置调试日志
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # RunPod 建议：如果是单机双卡，尝试禁用 P2P 和 IB 以排除硬件直连故障
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # 尝试网卡：如果 eth0 不行，请在这里改成 ens1（RunPod 常用高速网卡名）
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0" 

    print(f"进程 {rank} 正在初始化...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"进程 {rank} 初始化完成。")

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    try:
        setup(rank, world_size)
        
        # 简单的跨卡张量传输测试
        tensor = torch.ones(10, device=f"cuda:{rank}") * (rank + 1)
        print(f"Rank {rank} 发送前: {tensor[0].item()}")
        
        # 全局求和
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank} 接收后 (AllReduce): {tensor[0].item()}")
        
    finally:
        cleanup()

if __name__ == "__main__":
    world_size = 2 # 测试双卡
    print("开始双卡 NCCL 压力测试...")
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)