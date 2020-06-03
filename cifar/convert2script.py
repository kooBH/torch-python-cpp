import torch
import Net as Net
from torch.jit import trace

if __name__ == '__main__':
    net = Net.ResNet18()
    net.load_state_dict(torch.load('save/cifar10_resnet18_epoch10_state.pt'))
    net.eval()
    # 1 x 3 x 32 x 32
    example_input = torch.rand(1, 3, 32, 32)
    traced_model = trace(net, example_input)
    traced_model.save('traced-eval.pt')
