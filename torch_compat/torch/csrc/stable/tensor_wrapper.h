#pragma once

// Wrapper for torch/csrc/stable/tensor.h that avoids including tensor_inl.h
// to prevent multiple definition errors for scalar_type() when included in multiple .cu files
//
// The scalar_type() function is declared in tensor_struct.h and implemented in tensor_wrapper.cpp
// This ensures the function is only defined once, even when included in multiple translation units
//
// NOTE: We do NOT include library.h and ops.h here to avoid potential indirect includes of tensor.h
// These should be included separately where needed

#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

// Do NOT include library.h and ops.h here to avoid potential indirect includes
// Include them separately in files that need them

