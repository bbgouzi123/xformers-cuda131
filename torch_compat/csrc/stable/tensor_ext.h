#pragma once

// This file provides extensions to torch::stable::Tensor for PyTorch 2.9.1 compatibility
// We use a technique to inject methods into the Tensor class via a helper struct

// Use our wrapper instead of tensor.h to avoid multiple definition errors
// tensor_wrapper.h already includes tensor_struct.h and provides inline scalar_type()
#include <torch/csrc/stable/tensor_wrapper.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <vector>
#include <cstring>

// Forward declare the extension helper
namespace torch::stable::detail {
struct TensorExtHelper;
}

// Inject methods into Tensor class using a helper struct pattern
namespace torch::stable {
namespace detail {

struct TensorExtHelper {
  static Device get_device(const Tensor& t) {
    int32_t device_type;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(t.get(), &device_type));
    int32_t device_index;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_index(t.get(), &device_index));
    return Device(device_type, device_index);
  }

  static std::vector<int64_t> get_sizes(const Tensor& t) {
    int64_t dim_val;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(t.get(), &dim_val));
    std::vector<int64_t> result(dim_val);
    int64_t* sizes_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(t.get(), &sizes_ptr));
    if (sizes_ptr) {
      result.assign(sizes_ptr, sizes_ptr + dim_val);
    }
    return result;
  }

  static std::vector<int64_t> get_strides(const Tensor& t) {
    int64_t dim_val;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(t.get(), &dim_val));
    std::vector<int64_t> result(dim_val);
    int64_t* strides_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(t.get(), &strides_ptr));
    if (strides_ptr) {
      result.assign(strides_ptr, strides_ptr + dim_val);
    }
    return result;
  }

  template <typename T>
  static T* get_mutable_data_ptr(Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
  }

  template <typename T>
  static const T* get_const_data_ptr(const Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
  }
};

} // namespace detail
} // namespace torch::stable

// Use ADL (Argument-Dependent Lookup) to make these functions available
// when working with Tensor objects
namespace torch::stable {

// Free functions that can be called on Tensor objects
inline Device device(const Tensor& t) {
  return detail::TensorExtHelper::get_device(t);
}

inline std::vector<int64_t> sizes(const Tensor& t) {
  return detail::TensorExtHelper::get_sizes(t);
}

inline std::vector<int64_t> strides(const Tensor& t) {
  return detail::TensorExtHelper::get_strides(t);
}

template <typename T>
T* mutable_data_ptr(Tensor& t) {
  return detail::TensorExtHelper::get_mutable_data_ptr<T>(t);
}

// Const version for const tensors (needed for xf_packed_accessor)
template <typename T>
T* mutable_data_ptr(const Tensor& t) {
  return const_cast<T*>(detail::TensorExtHelper::get_mutable_data_ptr<T>(const_cast<Tensor&>(t)));
}

template <typename T>
const T* const_data_ptr(const Tensor& t) {
  return detail::TensorExtHelper::get_const_data_ptr<T>(t);
}

} // namespace torch::stable

// Note: We cannot directly add methods to the Tensor class in C++.
// xformers code will need to use the free functions above, or we need to
// modify xformers code to use these functions instead of member methods.
// Alternatively, we can create wrapper macros or modify the xformers code
// to call these functions.

