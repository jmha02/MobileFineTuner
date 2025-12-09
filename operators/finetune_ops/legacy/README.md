# Legacy NN Module Files

This directory contains several early NN implementations (such as `lora.h/cpp`, `lora_ops.h/cpp`, `attention.h/cpp`, `mlp.h/cpp`, `module.h`, `layers.h`, etc.). These files still depend on legacy Tensor interfaces and non-existent types like `BaseLayer`/`LayerConfig`/`GradTensorPtr`, and are kept for reference only. Direct inclusion in the build will result in compilation failures due to interface mismatches.

## Current Status
- ✅ `nn/lora_linear.h/cpp`: Migrated to modern TensorPtr/ops interface, in active use (still located in `finetune_ops/nn/`)
- ✅ `nn/embedding.h/cpp`: Migrated to modern interface, retained in `finetune_ops/nn/`
- ⚠️ `legacy/nn/lora.h/cpp`: Depends on removed BaseLayer/LayerConfig
- ⚠️ `legacy/nn/lora_ops.h/cpp`: Uses legacy `.data/.shape` Tensor API
- ⚠️ `legacy/nn/attention.h/cpp`, `legacy/nn/mlp.h/cpp`, `legacy/nn/module.h`, `legacy/nn/layers.h`: All are legacy interface examples
- ⚠️ `legacy/gpt2_finetune_model.h`: Depends on legacy interfaces like GradTensorPtr/Tokenizer

## Build Configuration
These legacy files have been excluded from FINETUNE_SOURCES in CMakeLists.txt and will not participate in compilation.

## Current Alternatives
- LoRA functionality: Use `nn/lora_linear.h` + `graph/lora_injector.h`
- Attention: Inline implementation in `graph/gpt2_model.cpp`
- Embedding: Use `nn/embedding.h` (migrated) or gpt2_model's embedding_lookup

## To Enable Legacy Files
Migration to modern Tensor/ops interface or implementation of missing base class framework is required first.
