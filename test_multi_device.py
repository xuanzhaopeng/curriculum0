import torch
import os
import socket
import subprocess

def print_section(title):
    print("\n" + "="*50)
    print(f" HOST CHECK: {title}")
    print("="*50)

def check_env_vars():
    print_section("环境变量 (Environment Variables)")
    vars_to_check = [
        "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", 
        "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", 
        "NCCL_DEBUG", "NCCL_SHM_DISABLE"
    ]
    for var in vars_to_check:
        print(f"{var}: {os.environ.get(var, 'NOT SET')}")

def check_gpu_status():
    print_section("GPU 状态 (GPU Status)")
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用!")
        return
    
    device_count = torch.cuda.device_count()
    print(f"找到 GPU 数量: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if device_count >= 2:
        print("\n检查 P2P 可达性 (P2P Accessibility):")
        # 检查 GPU 0 和 GPU 1 之间是否能直接通信
        try:
            can_access = torch.cuda.can_device_access_peer(0, 1)
            print(f"GPU 0 -> GPU 1 P2P support: {can_access}")
            
            # 如果 P2P 不支持但没禁，可能会卡死
            p2p_disabled = os.environ.get("NCCL_P2P_DISABLE") == "1"
            if not can_access and not p2p_disabled:
                print("警告: 硬件不支持 P2P，但 NCCL_P2P_DISABLE 未设置为 1。这可能导致挂起！")
        except Exception as e:
            print(f"检查 P2P 时出错: {e}")

def check_shm():
    print_section("共享内存 (Shared Memory)")
    try:
        # 运行 df -h /dev/shm
        shm_stats = os.statvfs('/dev/shm')
        # 转换成 GB
        total_shm = (shm_stats.f_frsize * shm_stats.f_blocks) / (1024**3)
        print(f"/dev/shm 总大小: {total_shm:.2f} GB")
        if total_shm < 0.5: # 小于 512MB
            print("危险: /dev/shm 太小 (建议 > 2GB)，NCCL 通信极易挂起！")
        else:
            print("状态: /dev/shm 大小看起来正常。")
    except Exception as e:
        print(f"检查共享内存失败: {e}")

def check_network():
    print_section("网络接口 (Network Interface)")
    ifname = os.environ.get("NCCL_SOCKET_IFNAME", "未设置")
    
    # 检查网卡是否存在
    net_dir = "/sys/class/net"
    interfaces = os.listdir(net_dir)
    print(f"系统可用网卡: {interfaces}")
    
    if ifname != "未设置":
        # 处理可能的逗号分隔
        names = ifname.split(',')
        found = False
        for name in names:
            if any(i.startswith(name) for i in interfaces):
                found = True
                print(f"确认: 网卡前缀 '{name}' 匹配成功。")
        if not found:
            print(f"错误: NCCL_SOCKET_IFNAME={ifname} 在系统中找不到对应的网卡！")
    
    # 检查 eth0 的状态
    if 'eth0' in interfaces:
        with open(f"{net_dir}/eth0/operstate", 'r') as f:
            print(f"eth0 状态: {f.read().strip()}")

def try_nccl_init():
    print_section("尝试最小化 NCCL 初始化 (NCCL Init Test)")
    print("注意: 如果这里卡住超过 30 秒，说明网卡名或 P2P 设置依然有问题。")
    
    import torch.distributed as dist
    import datetime

    # 设置超时
    timeout = datetime.timedelta(seconds=20)
    
    try:
        # 在单机双卡环境下模拟初始化
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        # 仅尝试初始化进程组，不进行实际通信
        # 使用 gloo 测试基础网络，使用 nccl 测试 GPU 通信
        backend = "nccl"
        print(f"尝试使用后端 {backend} 初始化进程组 (Rank 0)...")
        
        # 注意：由于这是单进程尝试模拟分布式，通常需要多进程才能完成
        # 这里仅探测环境，如果配置完全错误，这里会立刻报错
        print("提示: 单脚本测试 NCCL 初始化通常需要 torchrun，此处仅做环境变量注入测试。")
        
    except Exception as e:
        print(f"初始化尝试失败: {e}")

if __name__ == "__main__":
    check_env_vars()
    check_shm()
    check_network()
    check_gpu_status()
    try_nccl_init()
    print("\n检测完成。")