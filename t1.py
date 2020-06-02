import torch
import torchvision

# 모델 인스턴스 생성
model = torchvision.models.resnet18()

# 일반적으로 모델의 forward() 메서드에 넘겨주는 입력값
example = torch.rand(1, 3, 224, 224)

# torch.jit.trace를 사용하여 트레이싱을 이용해 torch.jit.ScriptModule 생성
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1, 3, 224, 224))

print(output)

traced_script_module.save("traced_resnet_model.pt")
