/**
 * @file tokenizer_bpe.h
 * @brief GPT-2 Byte-Level BPE Tokenizer (fully aligned with HuggingFace)
 * 
 * Implements standard GPT-2 BPE tokenization, supports:
 * - Loading vocab.json, merges.txt, special_tokens_map.json
 * - Byte-Level mapping (256 bytes → unicode)
 * - BPE merge rules (in rank order from merges.txt)
 * - Encoding/decoding consistent with HuggingFace transformers
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ops {

/**
 * @brief GPT-2 Byte-Level BPE Configuration
 */
struct BPEConfig {
    std::string vocab_path;           // Path to vocab.json
    std::string merges_path;          // Path to merges.txt
    std::string special_tokens_path;  // Path to special_tokens_map.json (optional)
    
    int eos_token_id = 50256;         // <|endoftext|>
    int bos_token_id = 50256;         // GPT-2 typically doesn't use separate BOS
    int pad_token_id = 50256;         // Recommended to reuse eos (HF default)
    int unk_token_id = 50256;         // Unknown token (rarely used in GPT-2)
    
    bool add_prefix_space = false;    // GPT-2 default doesn't add leading space
    
    BPEConfig() = default;
    
    // Load from directory (assumes standard naming)
    static BPEConfig from_pretrained(const std::string& model_dir) {
        BPEConfig cfg;
        cfg.vocab_path = model_dir + "/vocab.json";
        cfg.merges_path = model_dir + "/merges.txt";
        cfg.special_tokens_path = model_dir + "/special_tokens_map.json";
        return cfg;
    }
};

/**
 * @brief GPT-2 Byte-Level BPE Tokenizer
 * 
 * Aligned with HuggingFace GPT2Tokenizer:
 * - Byte→Unicode mapping (256 bytes → 256 characters, skipping unprintable ranges)
 * - BPE merging by rank order in merges.txt
 * - Special token: <|endoftext|> (id=50256)
 */
class GPT2BPETokenizer {
public:
    explicit GPT2BPETokenizer(const BPEConfig& config);
    ~GPT2BPETokenizer() = default;
    
    /**
     * @brief Load tokenizer assets (vocab/merges/special_tokens)
     */
    void load();
    
    /**
     * @brief Encode text to token IDs
     * @param text Raw text
     * @param add_special_tokens Whether to automatically add <|endoftext|> (default false, GPT-2 usually manual control)
     * @param max_length Maximum length (0 = no limit)
     * @param truncation Whether to truncate (when exceeding length)
     * @return token IDs
     */
    std::vector<int> encode(const std::string& text,
                            bool add_special_tokens = false,
                            int max_length = 0,
                            bool truncation = true);
    
    /**
     * @brief Decode token IDs to text
     * @param ids token IDs
     * @param skip_special_tokens Whether to skip special tokens (default false)
     * @return Original text
     */
    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = false);
    
    /**
     * @brief Batch encode (with padding/truncation)
     * @param texts List of texts
     * @param max_length Maximum length (0 = no limit)
     * @param padding Whether to pad ("max_length" | "longest" | "none")
     * @param truncation Whether to truncate
     * @return {input_ids, attention_mask}
     */
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> 
    batch_encode(const std::vector<std::string>& texts,
                 int max_length = 0,
                 const std::string& padding = "longest",
                 bool truncation = true);
    
    // Get special token IDs
    int get_eos_token_id() const { return config_.eos_token_id; }
    int get_bos_token_id() const { return config_.bos_token_id; }
    int get_pad_token_id() const { return config_.pad_token_id; }
    int get_unk_token_id() const { return config_.unk_token_id; }
    int get_vocab_size() const { return vocab_size_; }
    
    // Utilities
    std::string get_token_string(int token_id) const;

private:
    BPEConfig config_;
    
    // Byte to Unicode mapping (256 entries)
    std::unordered_map<uint8_t, std::string> byte_encoder_;    // byte → unicode UTF-8 char
    std::unordered_map<std::string, uint8_t> byte_decoder_;    // unicode UTF-8 char → byte
    
    // Vocab and reverse mapping
    std::unordered_map<std::string, int> vocab_;  // token → id
    std::unordered_map<int, std::string> id_to_token_;  // id → token
    int vocab_size_;
    
    // Merge rules (pair -> rank)
    std::unordered_map<std::string, int> bpe_ranks_;  // "a b" → rank
    
    // Special token cache
    std::unordered_map<std::string, int> special_tokens_;
    
    // Internal methods
    void build_byte_encoder();  // Build byte <-> unicode mapping
    void load_vocab();          // Load vocab.json
    void load_merges();         // Load merges.txt
    void load_special_tokens(); // Load special_tokens_map.json
    
    std::string bytes_to_unicode(const std::string& text);  // Text -> byte-level unicode
    std::string unicode_to_bytes(const std::string& unicode_text);  // Reverse mapping
    
    std::vector<std::string> bpe(const std::string& token);  // Apply BPE merges to single token
    std::pair<int, int> get_best_pair(const std::vector<std::string>& word);  // Find pair with highest rank
    std::vector<std::string> split_to_words(const std::string& text);  // Pre-split by whitespace
};

}  // namespace ops

