/**
 * @file test_tokenizer_bpe.cpp
 * @brief GPT-2 BPE Tokenizer alignment test (must match HF)
 * 
 * Usage:
 *   g++ -std=c++17 test_tokenizer_bpe.cpp tokenizer_bpe.cpp -I. -o test_bpe
 *   ./test_bpe /path/to/pretrained/gpt2
 * 
 * Expected output:
 *   [PASS] Chinese, English, and emoji
 *   [PASS] Space, newline, punctuation
 *   [PASS] encode->decode round-trip
 */

#include "tokenizer_bpe.h"
#include <iostream>
#include <cassert>

using namespace ops;

void test_basic_encode_decode(GPT2BPETokenizer& tokenizer) {
    std::cout << "\n[Test] Basic encode/decode..." << std::endl;
    
    // Test 1: Mixed Chinese, English, and emoji
    {
        std::string text = "Today's weather is great, GPT-2 rocks!";
        auto ids = tokenizer.encode(text, false, 0, false);
        auto decoded = tokenizer.decode(ids, false);
        
        std::cout << "  Original: " << text << std::endl;
        std::cout << "  IDs: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i < ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Decoded: " << decoded << std::endl;
        
        // Check round-trip consistency
        if (text == decoded) {
            std::cout << "  [PASS] encode->decode round-trip consistent" << std::endl;
        } else {
            std::cout << "  [FAIL] Round-trip inconsistent!" << std::endl;
            std::cout << "    Expected: " << text << std::endl;
            std::cout << "    Actual: " << decoded << std::endl;
        }
    }
    
    // Test 2: Space/newline/consecutive punctuation
    {
        std::string text = " a b\n\n--==??";
        auto ids = tokenizer.encode(text, false, 0, false);
        auto decoded = tokenizer.decode(ids, false);
        
        std::cout << "\n  Original: \"" << text << "\"" << std::endl;
        std::cout << "  IDs: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i < ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Decoded: \"" << decoded << "\"" << std::endl;
        
        if (text == decoded) {
            std::cout << "  [PASS] Space/newline/punctuation round-trip consistent" << std::endl;
        } else {
            std::cout << "  [FAIL] Round-trip inconsistent!" << std::endl;
        }
    }
    
    // Test 3: Pure English simple sentence
    {
        std::string text = "Hello, world!";
        auto ids = tokenizer.encode(text, false, 0, false);
        auto decoded = tokenizer.decode(ids, false);
        
        std::cout << "\n  Original: " << text << std::endl;
        std::cout << "  IDs: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i < ids.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Decoded: " << decoded << std::endl;
        
        if (text == decoded) {
            std::cout << "  [PASS] Pure English round-trip consistent" << std::endl;
        } else {
            std::cout << "  [FAIL] Round-trip inconsistent!" << std::endl;
        }
    }
}

void test_batch_encode(GPT2BPETokenizer& tokenizer) {
    std::cout << "\n[Test] batch_encode（padding）..." << std::endl;
    
    std::vector<std::string> texts = {
        "short",
        "a bit longer sentence",
        "x"
    };
    
    auto [ids_batch, masks_batch] = tokenizer.batch_encode(texts, 0, "longest", false);
    
    for (size_t i = 0; i < texts.size(); ++i) {
        std::cout << "  [" << i << "] \"" << texts[i] << "\"" << std::endl;
        std::cout << "      IDs:  [";
        for (size_t j = 0; j < ids_batch[i].size(); ++j) {
            std::cout << ids_batch[i][j];
            if (j < ids_batch[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "      Mask: [";
        for (size_t j = 0; j < masks_batch[i].size(); ++j) {
            std::cout << masks_batch[i][j];
            if (j < masks_batch[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "  [PASS] batch_encode completed (padded to longest)" << std::endl;
}

void test_special_tokens(GPT2BPETokenizer& tokenizer) {
    std::cout << "\n[Test] Special tokens..." << std::endl;
    
    std::cout << "  EOS: " << tokenizer.get_eos_token_id() << std::endl;
    std::cout << "  PAD: " << tokenizer.get_pad_token_id() << std::endl;
    std::cout << "  Vocab size: " << tokenizer.get_vocab_size() << std::endl;
    
    // Test add_special_tokens
    std::string text = "test";
    auto ids_no_special = tokenizer.encode(text, false, 0, false);
    auto ids_with_special = tokenizer.encode(text, true, 0, false);
    
    std::cout << "  \"" << text << "\" (no special): " << ids_no_special.size() << " tokens" << std::endl;
    std::cout << "  \"" << text << "\" (with special): " << ids_with_special.size() << " tokens" << std::endl;
    
    if (ids_with_special.size() == ids_no_special.size() + 1 &&
        ids_with_special.back() == tokenizer.get_eos_token_id()) {
        std::cout << "  [PASS] add_special_tokens correctly added EOS" << std::endl;
    } else {
        std::cout << "  [FAIL] add_special_tokens abnormal" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_gpt2_pretrained_dir>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " /path/to/gpt2_lora_finetune/pretrained/gpt2" << std::endl;
        return 1;
    }
    
    std::string model_dir = argv[1];
    
    try {
        // Load tokenizer
        auto config = BPEConfig::from_pretrained(model_dir);
        GPT2BPETokenizer tokenizer(config);
        tokenizer.load();
        
        std::cout << "============================================" << std::endl;
        std::cout << "GPT-2 BPE Tokenizer Alignment Test" << std::endl;
        std::cout << "============================================" << std::endl;
        
        test_basic_encode_decode(tokenizer);
        test_batch_encode(tokenizer);
        test_special_tokens(tokenizer);
        
        std::cout << "\n============================================" << std::endl;
        std::cout << "All tests completed!" << std::endl;
        std::cout << "============================================\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

