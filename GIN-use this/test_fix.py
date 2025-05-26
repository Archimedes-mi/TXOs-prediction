import torch
import torch.nn.functional as F

# 模拟原始情况 - 维度不匹配
output = torch.randn(64, 3)  # 模型输出 [64, 3]
target = torch.randn(192)    # 目标 [192]

print("输出形状:", output.shape)
print("目标形状:", target.shape)

try:
    # 这应该会失败，重现原始错误
    loss = F.mse_loss(output, target)
    print("直接计算成功，但不应该成功！")
except RuntimeError as e:
    print("原始错误:", e)

# 使用我们的修复方法
batch_size = output.size(0)
reshaped_target = target.view(batch_size, -1)

print("\n修复后:")
print("输出形状:", output.shape)
print("目标新形状:", reshaped_target.shape)

try:
    # 这应该会成功
    loss = F.mse_loss(output, reshaped_target)
    print("修复后的损失计算成功:", loss.item())
except RuntimeError as e:
    print("修复后仍有错误:", e) 