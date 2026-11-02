#include "vocab.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

static std::vector<std::string> split_words(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> words;
    std::string word;
    while (in >> word) words.push_back(word);
    return words;
}

Vocab::Vocab() {
    id_to_token_ = {"<pad>", "<unk>", "<bos>", "<eos>"};
    for (std::size_t i = 0; i < id_to_token_.size(); ++i)
        token_to_id_[id_to_token_[i]] = static_cast<int64_t>(i);
}

void Vocab::build(const std::vector<std::string>& sentences, std::size_t min_freq) {
    std::unordered_map<std::string, std::size_t> counts;
    for (const auto& sentence : sentences)
        for (const auto& word : split_words(sentence))
            if (word != "<pad>" && word != "<unk>" && word != "<bos>" && word != "<eos>")
                ++counts[word];

    std::vector<std::pair<std::string, std::size_t>> items(counts.begin(), counts.end());
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    id_to_token_ = {"<pad>", "<unk>", "<bos>", "<eos>"};
    token_to_id_.clear();
    for (std::size_t i = 0; i < id_to_token_.size(); ++i)
        token_to_id_[id_to_token_[i]] = static_cast<int64_t>(i);

    for (const auto& item : items) {
        if (item.second < min_freq) continue;
        const auto id = static_cast<int64_t>(id_to_token_.size());
        id_to_token_.push_back(item.first);
        token_to_id_[item.first] = id;
    }
}

std::vector<int64_t> Vocab::encode(const std::string& text, bool add_bos_eos) const {
    std::vector<int64_t> ids;
    if (add_bos_eos) ids.push_back(BOS);
    for (const auto& word : split_words(text)) {
        auto it = token_to_id_.find(word);
        ids.push_back(it == token_to_id_.end() ? UNK : it->second);
    }
    if (add_bos_eos) ids.push_back(EOS);
    return ids;
}

std::string Vocab::decode(const std::vector<int64_t>& ids) const {
    std::string result;
    for (auto id : ids) {
        if (id == EOS) break;
        if (id == PAD || id == BOS) continue;
        std::string token = (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size())
            ? id_to_token_[static_cast<std::size_t>(id)] : "<unk>";
        if (!result.empty()) result += ' ';
        result += token;
    }
    return result;
}

std::size_t Vocab::size() const { return id_to_token_.size(); }

void Vocab::save(const std::string& path) const {
    nlohmann::json j = id_to_token_;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write vocabulary: " + path);
    out << j.dump(2);
}

Vocab Vocab::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open vocabulary: " + path);
    nlohmann::json j;
    in >> j;
    Vocab vocab;
    vocab.id_to_token_ = j.get<std::vector<std::string>>();
    vocab.token_to_id_.clear();
    for (std::size_t i = 0; i < vocab.id_to_token_.size(); ++i)
        vocab.token_to_id_[vocab.id_to_token_[i]] = static_cast<int64_t>(i);
    if (vocab.id_to_token_.size() < 4 ||
        vocab.id_to_token_[0] != "<pad>" || vocab.id_to_token_[1] != "<unk>" ||
        vocab.id_to_token_[2] != "<bos>" || vocab.id_to_token_[3] != "<eos>")
        throw std::runtime_error("Invalid vocabulary special-token ordering: " + path);
    return vocab;
}
