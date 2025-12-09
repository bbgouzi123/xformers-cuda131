#pragma once

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

// IMPORTANT: Include tensor_wrapper.h FIRST before ops.h and library.h
// This ensures Tensor is properly defined before ops.h uses it
// Use wrapper instead of tensor.h to avoid multiple definition errors
#include <torch/csrc/stable/tensor_wrapper.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
// Include compatibility headers for PyTorch 2.9.1
// These will be found in torch_compat/torch/ directory via include_dirs
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor_ext.h>
#include <torch/headeronly/core/TensorAccessor.h>
// Include c10::guts types
#include <c10/util/TypeList.h>
#include <c10/util/Metaprogramming.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace {

// Add specializations for std::vector and std::string in detail namespace
// These are needed because PyTorch 2.9.1 stable API doesn't support them directly
namespace detail {

// Specialization for std::vector<int64_t> => StableIValue
template <>
struct FromImpl<std::vector<int64_t>> {
  static StableIValue call(const std::vector<int64_t>& val) {
    // Allocate heap memory and pass pointer (similar to std::optional)
    int64_t* data = new int64_t[val.size()];
    std::copy(val.begin(), val.end(), data);
    // Wrap in StableIValue* (similar to std::optional handling)
    StableIValue* wrapper = new StableIValue[2];
    wrapper[0] = from(reinterpret_cast<void*>(data));
    wrapper[1] = from(static_cast<int64_t>(val.size()));
    return from(wrapper);
  }
};

// Specialization for StableIValue => std::vector<int64_t>
template <>
struct ToImpl<std::vector<int64_t>> {
  static std::vector<int64_t> call(StableIValue val) {
    StableIValue* wrapper = to<StableIValue*>(val);
    if (wrapper == nullptr) {
      return std::vector<int64_t>();
    }
    int64_t* data_ptr = static_cast<int64_t*>(to<void*>(wrapper[0]));
    int64_t size = to<int64_t>(wrapper[1]);
    std::vector<int64_t> result(data_ptr, data_ptr + size);
    delete[] data_ptr;
    delete[] wrapper;
    return result;
  }
};

// Specialization for std::string => StableIValue
template <>
struct FromImpl<std::string> {
  static StableIValue call(const std::string& val) {
    char* data = new char[val.size()];
    std::copy(val.begin(), val.end(), data);
    StableIValue* wrapper = new StableIValue[2];
    wrapper[0] = from(reinterpret_cast<void*>(data));
    wrapper[1] = from(static_cast<int64_t>(val.size()));
    return from(wrapper);
  }
};

// Specialization for StableIValue => std::string
template <>
struct ToImpl<std::string> {
  static std::string call(StableIValue val) {
    StableIValue* wrapper = to<StableIValue*>(val);
    if (wrapper == nullptr) {
      return std::string();
    }
    void* data_ptr = to<void*>(wrapper[0]);
    int64_t size = to<int64_t>(wrapper[1]);
    std::string result(static_cast<const char*>(data_ptr), size);
    delete[] static_cast<char*>(data_ptr);
    delete[] wrapper;
    return result;
  }
};

} // namespace detail


template <class... T, std::size_t... I>
std::tuple<T...> unbox_to_tuple_impl(
    StableIValue* stack,
    std::index_sequence<I...>) {
  return std::make_tuple(
      to<std::remove_cv_t<std::remove_reference_t<T>>>(
          stack[I])...);
}

template <class... T>
std::tuple<T...> unbox_to_tuple(StableIValue* stack) {
  return unbox_to_tuple_impl<T...>(
      stack, std::make_index_sequence<sizeof...(T)>());
}

template <class... T, std::size_t... I>
void box_from_tuple_impl(
    StableIValue* stack,
    std::tuple<T...> vals,
    std::index_sequence<I...>) {
  ((stack[I] = from<
        std::remove_cv_t<std::remove_reference_t<T>>>(std::get<I>(vals))),
   ...);
}

template <class... T>
void box_from_tuple(StableIValue* stack, std::tuple<T...> vals) {
  box_from_tuple_impl<T...>(
      stack, vals, std::make_index_sequence<sizeof...(T)>());
}

template <
    typename ReturnType,
    typename ParameterTypeList,
    typename FuncT,
    FuncT* func>
struct boxer_impl {};

template <
    typename... ReturnTypes,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    std::tuple<ReturnTypes...>,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  void operator()(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    assert(num_args == sizeof...(ParameterTypes));
    assert(num_outputs == sizeof...(ReturnTypes));
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    auto res = std::apply(func, args);
    box_from_tuple<ReturnTypes...>(stack, res);
  }
};

template <
    typename ReturnType,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    ReturnType,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  void operator()(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    assert(num_args == sizeof...(ParameterTypes));
    assert(num_outputs == 1);
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    auto res = std::apply(func, args);
    stack[0] = from<ReturnType>(res);
    // box_from_tuple<std::tuple<ReturnType>>(stack, std::make_tuple(res));
  }
};

template <typename... ParameterTypes, typename FuncT, FuncT* func>
struct boxer_impl<
    void,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  void operator()(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    assert(num_args == sizeof...(ParameterTypes));
    assert(num_outputs == 0);
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    std::apply(func, args);
  }
};

template <typename FuncT, FuncT* func>
struct boxer {
  using FunctionTrait = c10::guts::infer_function_traits_t<FuncT>;

  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    boxer_impl<
        typename FunctionTrait::return_type,
        c10::guts::typelist::map_t<
            std::remove_reference_t,
            typename FunctionTrait::parameter_types>,
        FuncT,
        func>()(stack, num_args, num_outputs);
  }
};

} // namespace

#define XF_BOXED_FN(func)                                             \
  (boxer<                                                             \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      (func)>::boxed_fn)

#define XF_CUDA_CHECK(EXPR)                                                    \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    /* TODO Call stable version of c10::cuda::c10_cuda_check_implementation */ \
    if (__err != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(__err));                     \
    }                                                                          \
  } while (0)

#define XF_CUDA_KERNEL_LAUNCH_CHECK() XF_CUDA_CHECK(cudaGetLastError())

#define XF_CUDA_DRIVER_CHECK(EXPR)                   \
  do {                                               \
    const CUresult __err = EXPR;                     \
    if (__err != CUDA_SUCCESS) {                     \
      throw std::runtime_error("CUDA driver error"); \
    }                                                \
  } while (0)

cudaDeviceProp* xf_getCurrentDeviceProperties();

namespace {

cudaStream_t xf_getCurrentCUDAStream(
    torch::stable::accelerator::DeviceIndex index = -1) {
  // cudaStream_t ret;
  // TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(index, &ret));
  // return ret;
  return reinterpret_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .id());
}

template <typename dtype, size_t ndim>
auto xf_packed_accessor(const torch::stable::Tensor& t) {
  auto sizes_vec = torch::stable::sizes(t);
  auto strides_vec = torch::stable::strides(t);
  return torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<dtype, ndim>(
      torch::stable::mutable_data_ptr<dtype>(const_cast<torch::stable::Tensor&>(t)), 
      sizes_vec.data(), 
      strides_vec.data());
}

template <typename T>
constexpr __host__ __device__ inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

inline int32_t xf_get_layout(const torch::stable::Tensor& self) {
  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));
  return layout;
}

inline bool xf_is_sparse(const torch::stable::Tensor& self) {
  return xf_get_layout(self) != aoti_torch_layout_strided();
}

inline torch::stable::Tensor xf_view_dtype(
    const torch::stable::Tensor& self,
    torch::headeronly::ScalarType dtype) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      from(self), from(dtype)};
  // view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::view", "dtype", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_slice(
    const torch::stable::Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end) {
  const auto num_args = 5;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(dim),
      from(start),
      from(end),
      from(1)};
  // slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt?
  // end=None, SymInt step=1) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::slice", "Tensor", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_select(
    const torch::stable::Tensor& self,
    int64_t dim,
    int64_t index) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(dim),
      from(index)};
  // select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::select", "int", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_permute(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      from(self), from(dims)};
  // permute(Tensor(a) self, int[] dims) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::permute", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_contiguous(
    const torch::stable::Tensor& self,
    int32_t memory_format = aoti_torch_memory_format_contiguous_format()) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(memory_format),
  };
  // contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format)
  // -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::contiguous", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_zeros(
    std::vector<int64_t> size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  const auto num_args = 5;
  std::array<StableIValue, num_args> stack{
      from(size),
      from(dtype),
      from(std::nullopt),
      from(device),
      from(pin_memory)};
  // zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
  // Device? device=None, bool? pin_memory=None) -> Tensor
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::zeros", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

template <typename T>
inline torch::stable::Tensor xf_full(
    std::vector<int64_t> size,
    T fill_value,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      from(size),
      from(fill_value),
      from(dtype),
      from(std::nullopt),
      from(device),
      from(pin_memory)};
  // full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout?
  // layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::full", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_cumsum(
    const torch::stable::Tensor& self,
    int dim,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(dim),
      from(dtype)};
  // cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::cumsum", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_resize_(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size,
    int32_t memory_format = aoti_torch_memory_format_contiguous_format()) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(size),
      from(memory_format)};
  // resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat?
  // memory_format=None) -> Tensor(a!)
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::resize_", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

template <typename T>
inline T xf_item(const torch::stable::Tensor& self) {
  // Simplified implementation: use item op via dispatcher
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{from(self)};
  // item(Tensor self) -> Scalar
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::item", "", stack.data()));
  return to<T>(stack[0]);
}

// Make this function inline to avoid ODR violations
// The function calls scalar_type() which is defined in tensor_wrapper.cpp
inline size_t xf_element_size(const torch::stable::Tensor& self) {
#define RETURN_SIZEOF_IF_MATCHES_(cpp_type, dtype)                  \
  if (self.scalar_type() == torch::headeronly::ScalarType::dtype) { \
    return sizeof(cpp_type);                                        \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(RETURN_SIZEOF_IF_MATCHES_)
#undef RETURN_SIZEOF_IF_MATCHES_
  throw std::runtime_error("Unsupported dtype");
}

// Compatibility function for view operation
// Handles both explicit sizes and -1 for inferred dimension
inline torch::stable::Tensor xf_view(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size) {
  // Replace -1 with inferred size
  int64_t total_elements = 1;
  int64_t inferred_dim = -1;
  for (size_t i = 0; i < size.size(); ++i) {
    if (size[i] == -1) {
      inferred_dim = static_cast<int64_t>(i);
    } else {
      total_elements *= size[i];
    }
  }
  if (inferred_dim >= 0) {
    auto sizes_vec = torch::stable::sizes(self);
    int64_t self_elements = 1;
    for (size_t i = 0; i < sizes_vec.size(); ++i) {
      self_elements *= sizes_vec[i];
    }
    size[inferred_dim] = self_elements / total_elements;
  }
  // Use reshape via dispatcher
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      from(self), from(size)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(
      "aten::reshape", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

} // namespace

// Add view to torch::stable namespace for compatibility
namespace torch::stable {
inline Tensor view(const Tensor& self, std::vector<int64_t> size) {
  return xf_view(self, size);
}
} // namespace torch::stable
