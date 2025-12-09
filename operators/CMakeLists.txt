cmake_minimum_required(VERSION 3.10)
project(Operators VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# Production-Level Configuration Options
# ============================================================================
option(USE_NEW_AUTOGRAD_ENGINE "Use topological sort autograd engine" ON)
option(AUTOGRAD_DEBUG "Enable autograd debug output" OFF)
option(USE_MOBILE_OPTIMIZER "Enable mobile optimizer extensions" ON)
option(ENABLE_MEMORY_MODULE "Enable Memory module (parameter management/Zero optimization)" ON)
option(ENABLE_ACTIVATIONS_MODULE "Enable Activations module (gradient checkpointing/activation management)" ON)
option(USE_ARENA_ALLOCATOR "Use Arena memory allocator (prevents physical footprint growth)" OFF)
option(USE_BLAS "Enable BLAS acceleration (ON=Accelerate/OpenBLAS, OFF=pure C++)" OFF)
option(DISABLE_MEMORY_POOL "Disable custom memory pool, Tensor uses system malloc/free directly" OFF)
# Allow override via environment variable (e.g.: OPS_USE_BLAS=ON/OFF)
if(DEFINED ENV{OPS_USE_BLAS})
    string(TOUPPER "$ENV{OPS_USE_BLAS}" _OPS_USE_BLAS)
    if(_OPS_USE_BLAS STREQUAL "ON" OR _OPS_USE_BLAS STREQUAL "1" OR _OPS_USE_BLAS STREQUAL "TRUE")
        set(USE_BLAS ON CACHE BOOL "Enable BLAS via env" FORCE)
    elseif(_OPS_USE_BLAS STREQUAL "OFF" OR _OPS_USE_BLAS STREQUAL "0" OR _OPS_USE_BLAS STREQUAL "FALSE")
        set(USE_BLAS OFF CACHE BOOL "Disable BLAS via env" FORCE)
    endif()
endif()
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(ENABLE_PROFILING "Enable performance profiling" OFF)
option(ENABLE_GPT2_DEBUG_PRINT "Enable GPT-2 forward detailed debug printing" OFF)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compilation Options
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

if(AUTOGRAD_DEBUG)
    add_definitions(-DAUTOGRAD_DEBUG)
    message(STATUS "Autograd debug output enabled")
endif()

if(USE_NEW_AUTOGRAD_ENGINE)
    add_definitions(-DUSE_NEW_AUTOGRAD_ENGINE)
    message(STATUS "New autograd engine enabled (topological sort)")
endif()

if(USE_ARENA_ALLOCATOR)
    add_definitions(-DUSE_ARENA_ALLOCATOR)
    message(STATUS "Arena memory allocator enabled (prevents physical footprint growth)")
endif()

if(USE_BLAS)
    message(STATUS "BLAS enabled")
else()
    add_definitions(-DDISABLE_BLAS_COMPLETELY)
    message(STATUS "BLAS disabled (using pure C++ memory-first kernels)")
endif()

if(DISABLE_MEMORY_POOL)
    add_definitions(-DDISABLE_MEMORY_POOL)
    message(STATUS "Custom memory pool disabled (Tensor uses system malloc/free for RSS growth diagnosis)")
endif()

# Control GPT-2 debug printing
if(ENABLE_GPT2_DEBUG_PRINT)
    add_definitions(-DENABLE_GPT2_DEBUG_PRINT)
    message(STATUS "GPT-2 forward debug printing: ON")
else()
    message(STATUS "GPT-2 forward debug printing: OFF")
endif()

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    message(STATUS "Performance profiling enabled")
endif()

# ============================================================================
# Include Directories (added finetune_ops and opt_ops root directories)
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops
    ${CMAKE_CURRENT_SOURCE_DIR}/opt_ops
)

# =========================================================================
# Source Files List
# =========================================================================

set(FINETUNE_SOURCES
    # Core system files
    finetune_ops/core/tensor.cpp
    finetune_ops/core/ops.cpp
    finetune_ops/core/autograd_engine.cpp
    finetune_ops/core/backward_functions.cpp
    finetune_ops/core/logger.cpp
    finetune_ops/core/memory_manager.cpp
    finetune_ops/core/step_arena.cpp
    finetune_ops/core/utils.cpp
    finetune_ops/core/lm_loss.cpp
    finetune_ops/core/memory_efficient_attention.cpp
    finetune_ops/core/performance_monitor.cpp
    
    # GPT-2 fine-tuning core files
    finetune_ops/core/tokenizer_bpe.cpp
    finetune_ops/graph/gemma_model.cpp
    finetune_ops/graph/gemma_lora_injector.cpp
    finetune_ops/core/tokenizer_gemma.cpp
    finetune_ops/graph/safetensors_loader.cpp
    finetune_ops/graph/gpt2_model.cpp
    finetune_ops/graph/lora_injector.cpp
    
    # Phase B: Data loading and training
    finetune_ops/data/wikitext2_dataset.cpp
    finetune_ops/nn/lora_linear.cpp
    finetune_ops/optim/optimizer.cpp
    finetune_ops/optim/adam.cpp
    finetune_ops/optim/trainer.cpp
    
    # Phase C: LoRA save/load
    finetune_ops/graph/lora_saver.cpp
    finetune_ops/optim/gemma_trainer.cpp
)

# OPT_SOURCES temporarily fully disabled (not needed for Phase A/B, avoid dependency issues)
set(OPT_SOURCES
    opt_ops/energy/power_monitor.cpp
    opt_ops/sharding/parameter_sharder.cpp
)

# =========================================================================
# Create Static Library
# =========================================================================
add_library(operators STATIC
    ${FINETUNE_SOURCES}
    ${OPT_SOURCES}
)

# =============================================================================
# Modular Targets (finetune_ops / opt_ops)
# =============================================================================
# Using new directory structure (no longer referencing modules subdirectory)

# ============================================================================
# Link Library Detection (Accelerate/BLAS)
# ============================================================================
if(USE_BLAS)
    if(APPLE)
        # macOS: Use Accelerate Framework
        find_library(ACCELERATE_LIBRARY Accelerate)
        if(ACCELERATE_LIBRARY)
            target_link_libraries(operators PUBLIC ${ACCELERATE_LIBRARY})
            add_definitions(-DUSE_BLAS)
            message(STATUS "Linked Accelerate Framework: ${ACCELERATE_LIBRARY}")
        else()
            message(WARNING "Accelerate not found, falling back to pure C++ implementation")
            add_definitions(-DDISABLE_BLAS_COMPLETELY)
        endif()
    else()
        # Linux/Windows: Try OpenBLAS or MKL
        find_package(BLAS)
        if(BLAS_FOUND)
            target_link_libraries(operators PUBLIC ${BLAS_LIBRARIES})
            add_definitions(-DUSE_BLAS)
            message(STATUS "Linked BLAS: ${BLAS_LIBRARIES}")
        else()
            message(WARNING "BLAS not found, falling back to pure C++ implementation")
            add_definitions(-DDISABLE_BLAS_COMPLETELY)
        endif()
    endif()
else()
    message(STATUS "BLAS disabled, using pure C++ implementation")
endif()

# ============================================================================
# Thread Support
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(operators PUBLIC Threads::Threads)

# ============================================================================
# Output Configuration
# ============================================================================
set_target_properties(operators PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    VERSION ${PROJECT_VERSION}
)

# ============================================================================
# Install Rules
# ============================================================================
install(TARGETS operators
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/operators
    FILES_MATCHING PATTERN "*.h"
)

# =============================================================================
# Modular Targets (finetune_ops / opt_ops)
# =============================================================================
# Using new directory structure (no longer referencing modules subdirectory)

# ============================================================================
# Test Targets
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    
    # Autograd engine test
    if(USE_NEW_AUTOGRAD_ENGINE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_autograd_engine.cpp)
            add_executable(test_autograd_engine
                ../tests/test_autograd_engine.cpp
            )
            target_link_libraries(test_autograd_engine operators)
            add_test(NAME AutogradEngine COMMAND test_autograd_engine)
        endif()
    endif()
    
    # Optimizer test
    if(USE_MOBILE_OPTIMIZER)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/optim/comprehensive_optimizer_test.cpp)
            add_executable(test_optimizer
                optim/comprehensive_optimizer_test.cpp
            )
            target_link_libraries(test_optimizer operators)
            add_test(NAME Optimizer COMMAND test_optimizer)
        endif()
    endif()
    
    # Memory module test
    if(ENABLE_MEMORY_MODULE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_memory_module.cpp)
            add_executable(test_memory_module
                tests/test_memory_module.cpp
            )
            target_link_libraries(test_memory_module operators)
            add_test(NAME MemoryModule COMMAND test_memory_module)
        endif()
    endif()
    
    # Activations module test
    if(ENABLE_ACTIVATIONS_MODULE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_activations_module.cpp)
            add_executable(test_activations_module
                tests/test_activations_module.cpp
            )
            target_link_libraries(test_activations_module operators)
            add_test(NAME ActivationsModule COMMAND test_activations_module)
        endif()
    endif()
    
    # === Phase B Test Programs ===
    
    # Minimal training loop
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_train_minimal.cpp)
        add_executable(test_train_minimal
            finetune_ops/optim/test_train_minimal.cpp
        )
        target_link_libraries(test_train_minimal operators)
    endif()
    
    # Backward gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_backward_sanity.cpp)
        add_executable(test_backward_sanity
            finetune_ops/optim/test_backward_sanity.cpp
        )
        target_link_libraries(test_backward_sanity operators)
    endif()
    
    # Simple gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_simple_grad.cpp)
        add_executable(test_simple_grad
            finetune_ops/optim/test_simple_grad.cpp
        )
        target_link_libraries(test_simple_grad operators)
    endif()
    
    # QKT * softmax backward unit test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_qkt_softmax_grad.cpp)
        add_executable(test_qkt_softmax_grad
            finetune_ops/optim/test_qkt_softmax_grad.cpp
        )
        target_link_libraries(test_qkt_softmax_grad operators)
    endif()
    
    # RMSNorm forward+backward small-dimension unit test (T4)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_rmsnorm_unit.cpp)
        add_executable(test_rmsnorm_unit
            finetune_ops/optim/test_rmsnorm_unit.cpp
        )
        target_link_libraries(test_rmsnorm_unit operators)
        add_test(NAME RMSNormUnit COMMAND test_rmsnorm_unit)
    endif()
    
    # repeat_kv + softmax backward unit test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_repeat_kv_softmax_grad.cpp)
        add_executable(test_repeat_kv_softmax_grad
            finetune_ops/optim/test_repeat_kv_softmax_grad.cpp
        )
        target_link_libraries(test_repeat_kv_softmax_grad operators)
    endif()
    
    # T3: single-layer attention-only backward microtest
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_attention_single_layer_backward.cpp)
        add_executable(test_attention_single_layer_backward
            finetune_ops/optim/test_attention_single_layer_backward.cpp
        )
        target_link_libraries(test_attention_single_layer_backward operators)
    endif()
    
    # LoRALinear gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_lora_grad.cpp)
        add_executable(test_lora_grad
            finetune_ops/optim/test_lora_grad.cpp
        )
        target_link_libraries(test_lora_grad operators)
    endif()

    # Standalone LoRALinear forward/backward numerical alignment test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/nn/test_lora_linear.cpp)
        add_executable(test_lora_linear
            finetune_ops/nn/test_lora_linear.cpp
        )
        target_link_libraries(test_lora_linear operators)
        add_test(NAME LoRALinearUnit COMMAND test_lora_linear)
    endif()
    
    # Cross-Entropy gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_ce_grad.cpp)
        add_executable(test_ce_grad
            finetune_ops/optim/test_ce_grad.cpp
        )
        target_link_libraries(test_ce_grad operators)
    endif()
    
    # 10-step training convergence validation
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_10step_convergence.cpp)
        add_executable(test_10step_convergence
            finetune_ops/optim/test_10step_convergence.cpp
        )
        target_link_libraries(test_10step_convergence operators)
    endif()
    
    # LoRA Round-Trip test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_lora_roundtrip.cpp)
        add_executable(test_lora_roundtrip
            finetune_ops/graph/test_lora_roundtrip.cpp
        )
        target_link_libraries(test_lora_roundtrip operators)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_config.cpp)
        add_executable(test_gemma_config
            finetune_ops/graph/test_gemma_config.cpp
        )
        target_link_libraries(test_gemma_config operators)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_forward.cpp)
        add_executable(test_gemma_forward
            finetune_ops/graph/test_gemma_forward.cpp
        )
        target_link_libraries(test_gemma_forward operators)
    endif()
    
    # Baseline GPT-2 training (full parameter fine-tuning comparison)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_baseline_10step.cpp)
        add_executable(test_baseline_10step
            finetune_ops/optim/test_baseline_10step.cpp
        )
        target_link_libraries(test_baseline_10step operators)
    endif()
    
    # Fixed batch comparison test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_fixed_batch_compare.cpp)
        add_executable(test_fixed_batch_compare
            finetune_ops/optim/test_fixed_batch_compare.cpp
        )
        target_link_libraries(test_fixed_batch_compare operators)
    endif()
    
    message(STATUS "Test targets enabled")
endif()

# ============================================================================
# Main Program: gpt2_lora_finetune
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/main.cpp)
    add_executable(gpt2_lora_finetune
        ../gpt2_lora_finetune/main.cpp
    )
    target_link_libraries(gpt2_lora_finetune operators)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/train_lora_gemma.cpp)
    add_executable(train_lora_gemma
        finetune_ops/optim/train_lora_gemma.cpp
    )
    target_link_libraries(train_lora_gemma operators)
endif()

# MMLU eval
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/eval_mmlu.cpp)
    add_executable(eval_mmlu
        ../gpt2_lora_finetune/eval_mmlu.cpp
        ../gpt2_lora_finetune/mmlu/mmlu_runner.cpp
    )
    target_include_directories(eval_mmlu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(eval_mmlu operators)
endif()

# WikiText-2 PPL eval
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/eval_ppl.cpp)
    add_executable(eval_ppl
        ../gpt2_lora_finetune/eval_ppl.cpp
    )
    target_include_directories(eval_ppl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(eval_ppl operators)
endif()

# Quick PPL eval
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/quick_eval_ppl.cpp)
    add_executable(quick_eval_ppl
        ../gpt2_lora_finetune/quick_eval_ppl.cpp
    )
    target_include_directories(quick_eval_ppl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(quick_eval_ppl operators)
endif()

# Quick PPL eval with LoRA support
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/quick_eval_lora.cpp)
    add_executable(quick_eval_lora
        ../gpt2_lora_finetune/quick_eval_lora.cpp
    )
    target_include_directories(quick_eval_lora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
    target_link_libraries(quick_eval_lora operators)
endif()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "============================================")
message(STATUS "Operators Framework Configuration Summary v${PROJECT_VERSION}")
message(STATUS "============================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Core Modules:")
message(STATUS "  - Autograd Engine: ${USE_NEW_AUTOGRAD_ENGINE}")
message(STATUS "  - Mobile Optimizer: ${USE_MOBILE_OPTIMIZER}")
message(STATUS "Advanced Modules:")
message(STATUS "  - Memory Module: ${ENABLE_MEMORY_MODULE}")
message(STATUS "  - Activations Module: ${ENABLE_ACTIVATIONS_MODULE}")
message(STATUS "Development Tools:")
message(STATUS "  - Build Tests: ${BUILD_TESTS}")
message(STATUS "  - Profiling: ${ENABLE_PROFILING}")
message(STATUS "============================================")
message(STATUS "")
