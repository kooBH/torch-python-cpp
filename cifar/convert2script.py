import torch
import Net as Net

if __name__ == '__main__':
    net = Net.ResNet18()
    example_input = torch.rand(1, 3, 32, 32)
    print('loading pre-trainded model..')
    net.load_state_dict(torch.load('save/cifar10_resnet18_epoch1_state.pt'))
    print('eval')
    net.eval()

    print('tracing')
    traced_model = torch.jit.trace(net, example_input)
    print('saving')
    traced_model.save('traced-state2.pt')

