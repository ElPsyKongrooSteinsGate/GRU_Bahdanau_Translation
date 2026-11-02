#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

class Vocab {
public:
    static constexpr int64_t PAD = 0;
    static constexpr int64_t UNK = 1;
    static constexpr int64_t BOS = 2;
    static constexpr int64_t EOS = 3;

    Vocab();
    void build(const std::vector<std::string>& sentences, std::size_t min_freq = 1);
    std::vector<int64_t> encode(const std::string& text, bool add_bos_eos) const;
    std::string decode(const std::vector<int64_t>& ids) const;
    std::size_t size() const;
    void save(const std::string& path) const;
    static Vocab load(const std::string& path);

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int64_t> token_to_id_;
};
