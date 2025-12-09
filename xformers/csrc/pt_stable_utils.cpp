#include <deque>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// Define XF_CUDA_CHECK macro locally to avoid including the entire pt_stable_utils.h
// which might indirectly include tensor_inl.h and cause multiple definition errors
#define XF_CUDA_CHECK(EXPR)                                                    \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    if (__err != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(__err));                     \
    }                                                                          \
  } while (0)

// Forward declaration to avoid including pt_stable_utils.h
// which includes many headers that might cause multiple definition errors

namespace {

std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    int num_gpus;
    XF_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    device_flags.resize(num_gpus);
    device_properties.resize(num_gpus);
    return true;
  }();
}

void initDeviceProperty(torch::stable::accelerator::DeviceIndex device_index) {
  cudaDeviceProp device_prop{};
  XF_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // namespace

cudaDeviceProp* xf_getCurrentDeviceProperties() {
  initCUDAContextVectors();
  torch::stable::accelerator::DeviceIndex device =
      torch::stable::accelerator::getCurrentDeviceIndex();
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}
