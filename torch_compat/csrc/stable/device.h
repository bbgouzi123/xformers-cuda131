#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <cstring>
#include <cstdlib>

namespace torch::stable {

// Device class for PyTorch 2.9.1 compatibility
// This is a simplified version that matches xformers' usage
class Device {
 public:
  Device() : device_type_(aoti_torch_device_type_cpu()), device_index_(-1) {}
  
  explicit Device(int32_t device_type, DeviceIndex device_index = -1)
      : device_type_(device_type), device_index_(device_index) {}
  
  explicit Device(const char* device_string) {
    // Simple parsing for "cuda:0", "cpu", etc.
    if (std::strncmp(device_string, "cuda", 4) == 0) {
      device_type_ = aoti_torch_device_type_cuda();
      if (std::strlen(device_string) > 5) {
        device_index_ = std::atoi(device_string + 5);
      } else {
        device_index_ = 0;
      }
    } else if (std::strncmp(device_string, "cpu", 3) == 0) {
      device_type_ = aoti_torch_device_type_cpu();
      device_index_ = -1;
    } else {
      device_type_ = aoti_torch_device_type_cpu();
      device_index_ = -1;
    }
  }

  DeviceIndex index() const { return device_index_; }
  int32_t type() const { return device_type_; }
  
  bool operator==(const Device& other) const {
    return device_type_ == other.device_type_ && device_index_ == other.device_index_;
  }
  
  bool operator!=(const Device& other) const {
    return !(*this == other);
  }

 private:
  int32_t device_type_;
  DeviceIndex device_index_;
};

} // namespace torch::stable

