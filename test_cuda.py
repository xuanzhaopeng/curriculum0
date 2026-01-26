import torch
print(f"是否有可用CUDA: {torch.cuda.is_available()}")
try:
    for i in range(torch.cuda.device_count()):
        print(f"尝试连接设备 {i}: {torch.cuda.get_device_name(i)}")
        # 尝试在该卡上创建一个张量
        test_tensor = torch.zeros(1).to(f"cuda:{i}")
        print(f"设备 {i} 连接成功！")
except Exception as e:
    print(f"连接失败，报错信息: {e}")