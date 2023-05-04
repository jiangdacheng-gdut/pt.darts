import torch
import torch.nn as nn
import torch.quantization
from torchviz import make_dot

# 加载模型
model = torch.load('/root/dacheng/pt.darts/searchs/cifar10/best.pth.tar')

# Step 1 统计模型参数
# 统计模型参数数量和大小
total_params = sum(p.numel() for p in model.parameters())
total_size = sum(p.numel() * p.element_size() for p in model.parameters())

# 打印统计结果
print(f'Total parameters: {total_params}')
print(f'Total size (bytes): {total_size}')
print(f'Total size (MB): {total_size / (1024 * 1024)}')

# # Step 2 输出模型结构
# # 遍历模型的参数并打印输出
# for name, param in model.named_parameters():
#     print(name)
#     print(param.size())
#     print("------------")

# Step 3 结构可视化
# 可视化失败，应该是因为输入值格式错误，后续再进行修改
# x = torch.randn(1, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
# x = x.cuda()
# y = model(x)

# MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "png"

# # 指定文件生成的文件夹
# MyConvNetVis.directory = "/root/dacheng/proxylessnas"

# # 生成文件
# MyConvNetVis.save("/root/dacheng/proxylessnas/my_model.jpg")