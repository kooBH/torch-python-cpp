#include <torch/script.h> // 필요한 단 하나의 헤더파일.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::jit::script::Module module;

  try {
    // torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
    module = torch::jit::load("../traced_resnet_model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

// 입력값 벡터를 생성합니다.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));

// 모델을 실행한 뒤 리턴값을 텐서로 변환합니다.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}
