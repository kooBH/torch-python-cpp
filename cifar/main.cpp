#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // opencv input/output
#include <opencv2/imgproc/imgproc.hpp> // cvtColor
#include <opencv2/imgcodecs.hpp>

// CIFAR-10 classes
const std::vector<std::string> classes{"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

at::Tensor imageToTensor(cv::Mat & image);
void predict(torch::jit::script::Module & module, cv::Mat & image);

// Adapted from https://github.com/goldsborough/examples/blob/cpp/cpp/mnist/mnist.cpp#L106
// Parameters: means and stddevs lists size must match number of channels for input Tensor
//             e.g., means and stddevs must be of size C for Tensor of shape 1 x C x H x W
struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(const std::initializer_list<float> & means, const std::initializer_list<float> & stddevs)
        : means_(insertValues(means)), stddevs_(insertValues(stddevs)) {}
    std::list<torch::Tensor> insertValues(const std::initializer_list<float> & values) {
        std::list<torch::Tensor> tensorList;
        for (auto val : values) {
            tensorList.push_back(torch::tensor(val));
        }
        return tensorList;
    }
    torch::Tensor operator()(torch::Tensor input) {
      std::list<torch::Tensor>::iterator meanIter = means_.begin();
      std::list<torch::Tensor>::iterator stddevIter = stddevs_.begin();
      //  Substract each channel's mean and divide by stddev in place
      for (int i{0}; meanIter != means_.end() && stddevIter != stddevs_.end(); ++i, ++meanIter, ++stddevIter){
          //std::cout << "Mean: " << *meanIter << " Stddev: " << *stddevIter << std::endl;
          //std::cout << input[0][i] << std::endl;
          input[0][i].sub_(*meanIter).div_(*stddevIter);
      }
    return input;
    }

    std::list<torch::Tensor> means_, stddevs_;
};


int main(int argc, const char* argv[]) {
  torch::jit::script::Module module;

  try {
    // torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
//    module = torch::jit::load("../save/cifar10_resnet18_epoch10_state.pt");
    module = torch::jit::load("../traced-state.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n"<<e.what();
    return -1;
  }

  std::cout << "ok\n";

  // Input
  cv::Mat image;
  image = cv::imread("../pic/dog1.jpg", cv::IMREAD_COLOR);
  // Resize image
  cv::resize( image, image, cv::Size( 32,32 ));

  // Check for invalid input
  if(! image.data ) {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  // Check for cuda
  if (torch::cuda::is_available()) {
    std::cout << "Moving model to GPU" << std::endl;
    module.to(at::kCUDA);
  }

  predict(module, image);

  return 0;
}

at::Tensor imageToTensor(cv::Mat & image) {
    // BGR to RGB, which is what our network was trained on
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    // Convert Mat image to tensor 1 x C x H x W
    at::Tensor tensorImage = torch::from_blob(image.data, {1, image.rows, image.cols, image.channels()}, at::kByte);

    // Normalize tensor values from [0, 255] to [0, 1]
    tensorImage = tensorImage.toType(at::kFloat);
    tensorImage = tensorImage.div_(255);

    // Transpose the image for [channels, rows, columns] format of torch tensor
    tensorImage = at::transpose(tensorImage, 1, 2);
    tensorImage = at::transpose(tensorImage, 1, 3);
    return tensorImage; // 1 x C x H x W
}

void predict(torch::jit::script::Module & module, cv::Mat & image) {
    at::Tensor tensorImage{imageToTensor(image)};

    // Normalize
    struct Normalize normalizeChannels({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
    tensorImage = normalizeChannels(tensorImage);
    std::cout << "Image tensor shape: " << tensorImage.sizes() << std::endl;

    // Move tensor to CUDA memory
    tensorImage = tensorImage.to(at::kCUDA);
    // Forward pass
    std::cout<<"forward"<<std::endl;
    at::Tensor result = module.forward({tensorImage}).toTensor();
    auto maxResult = result.max(1);
    auto maxIndex = std::get<1>(maxResult).item<float>();
    auto maxOut = std::get<0>(maxResult).item<float>();
    std::cout << "Predicted: " << classes[maxIndex] << " | " << maxOut << std::endl;
}

