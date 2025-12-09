// Implementation file for tensor_wrapper.h
// This file contains the non-inline definition of scalar_type() to avoid multiple definition errors
//
// IMPORTANT: Do NOT include tensor_wrapper.h here, as it may indirectly include tensor.h
// which would cause multiple definition errors. Only include the minimal headers needed.

#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::stable {
using torch::headeronly::ScalarType;

// Non-inline definition of scalar_type() to avoid multiple definition errors
// This will be compiled only once and linked with all translation units
ScalarType Tensor::scalar_type() const {
  int32_t dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(ath_.get(), &dtype));
  return to<ScalarType>(from(dtype));
}
} // namespace torch::stable

