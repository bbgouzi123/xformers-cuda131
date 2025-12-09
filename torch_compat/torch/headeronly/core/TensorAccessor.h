#pragma once

#include <ATen/core/TensorAccessor.h>

namespace torch::headeronly {

// Wrapper for ATen TensorAccessor to match xformers expectations
// This is a compatibility layer for PyTorch 2.9.1
template <typename T, size_t N>
using HeaderOnlyGenericPackedTensorAccessor = at::GenericPackedTensorAccessor<T, N>;

} // namespace torch::headeronly

