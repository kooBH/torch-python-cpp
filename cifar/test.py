import torch
import torchvision
import torchvision.transforms as transforms
import Net as Net

if __name__ == '__main__':
    transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                    shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    #dataiter = iter(testloader)
    #images, labels = dataiter.next()

    # 이미지를 출력합니다.
    #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net.ResNet18()
    net.load_state_dict(torch.load('save/cifar_state.pt'))
    net.cuda()

    #outputs = net(images)
    #_, predicted = torch.max(outputs, 1)
    #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                          for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
