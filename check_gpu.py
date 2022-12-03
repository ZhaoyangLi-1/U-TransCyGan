import torch

print("GPU Count: {}".format(torch.cuda.device_count))
print("GPU {}: {}".format(torch.cuda.get_device_name(0), torch.cuda.memory_summary(0)))
print("GPU {}: {}".format(torch.cuda.get_device_name(1), torch.cuda.memory_summary(1)))
print("GPU {}: {}".format(torch.cuda.get_device_name(2), torch.cuda.memory_summary(2)))
print("GPU {}: {}".format(torch.cuda.get_device_name(3), torch.cuda.memory_summary(3)))
print("GPU {}: {}".format(torch.cuda.get_device_name(4), torch.cuda.memory_summary(4)))
print("GPU {}: {}".format(torch.cuda.get_device_name(5), torch.cuda.memory_summary(5)))
print("GPU {}: {}".format(torch.cuda.get_device_name(6), torch.cuda.memory_summary(6)))
print("GPU {}: {}".format(torch.cuda.get_device_name(6), torch.cuda.memory_summary(7)))

