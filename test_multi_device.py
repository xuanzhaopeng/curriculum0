import torch
import torch.distributed as dist
import os

os.environ["NCCL_DEBUG"] = "INFO"

def test():
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return

    # 准备初始化
    dist.init_process_group(backend="nccl", rank=0, world_size=1) 
    print("Rank 0 已启动")
    
    # 真正的双卡测试（模拟两张卡握手）
    # 如果你是单脚本运行双卡，建议用下面这种最直接的测试：
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    a = torch.ones(100).to(device0)
    b = torch.ones(100).to(device1)
    
    print("正在尝试跨卡同步...")
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)
    print("跨卡同步成功！说明显卡驱动和基础 CUDA 通信正常。")

if __name__ == "__main__":
    test()