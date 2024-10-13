from torchsummary import summary
from thop import profile
from torchstat import stat
from networks.MeshSegNet import MeshSegNet
import torch

model = MeshSegNet().cuda()
data_shape = (2, 24, 300)
summary(model, data_shape)
flops, params = profile(model, inputs=(torch.rand(1, 2, 24,300).cuda(),))
print('flops:{}'.format(flops))
print('params:{}'.format(params))