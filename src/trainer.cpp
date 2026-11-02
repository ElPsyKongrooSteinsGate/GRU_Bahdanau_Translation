#include "trainer.h"
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct EncodedPair { std::vector<int64_t> src, tgt; };

static torch::Tensor pad_batch(const std::vector<std::vector<int64_t>>& sequences, int64_t pad,
                               const torch::Device& device) {
    int64_t max_len = 1;
    for (const auto& s : sequences) max_len = std::max<int64_t>(max_len, static_cast<int64_t>(s.size()));
    auto out = torch::full({static_cast<int64_t>(sequences.size()), max_len}, pad,
        torch::TensorOptions().dtype(torch::kLong).device(device));
    for (std::size_t i = 0; i < sequences.size(); ++i) {
        auto row = torch::tensor(sequences[i], torch::TensorOptions().dtype(torch::kLong).device(device));
        out[static_cast<int64_t>(i)].narrow(0, 0, row.size(0)).copy_(row);
    }
    return out;
}

void train_model(const std::string& db_path, const std::string& model_dir, const TrainConfig& cfg) {
    TranslationDB db(db_path);
    auto rows = db.load_all();
    if (rows.empty()) throw std::runtime_error("Training database is empty. Import parallel sentences first.");

    std::vector<std::string> src_texts, tgt_texts;
    for (const auto& p : rows) { src_texts.push_back(p.source); tgt_texts.push_back(p.target); }

    Vocab src_vocab, tgt_vocab;
    src_vocab.build(src_texts);
    tgt_vocab.build(tgt_texts);
    fs::create_directories(model_dir);
    src_vocab.save((fs::path(model_dir) / "source_vocab.json").string());
    tgt_vocab.save((fs::path(model_dir) / "target_vocab.json").string());

    std::vector<EncodedPair> data;
    for (const auto& p : rows) {
        auto s = src_vocab.encode(p.source, true);
        auto t = tgt_vocab.encode(p.target, true);
        if (static_cast<int64_t>(s.size()) <= cfg.max_length &&
            static_cast<int64_t>(t.size()) <= cfg.max_length)
            data.push_back({std::move(s), std::move(t)});
    }
    if (data.empty()) throw std::runtime_error("No examples remain after max_length filtering.");

    const auto device = torch::Device(torch::kCPU);
    Seq2Seq model(src_vocab.size(), tgt_vocab.size(), cfg.embedding_dim,
                  cfg.hidden_dim, cfg.attention_dim, Vocab::PAD);
    model->to(device);
    model->train();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(cfg.learning_rate));

    std::mt19937 rng(12345);
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);

    for (int64_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double total_loss = 0.0;
        std::size_t batches = 0;
        for (std::size_t begin = 0; begin < order.size(); begin += static_cast<std::size_t>(cfg.batch_size)) {
            std::vector<std::vector<int64_t>> src_batch, tgt_batch;
            const auto end = std::min(order.size(), begin + static_cast<std::size_t>(cfg.batch_size));
            for (std::size_t i = begin; i < end; ++i) {
                src_batch.push_back(data[order[i]].src);
                tgt_batch.push_back(data[order[i]].tgt);
            }
            auto source = pad_batch(src_batch, Vocab::PAD, device);
            auto target = pad_batch(tgt_batch, Vocab::PAD, device);

            optimizer.zero_grad();
            auto logits = model->forward(source, target, cfg.teacher_forcing_ratio);
            auto labels = target.narrow(1, 1, target.size(1) - 1);
            auto loss = torch::nn::functional::cross_entropy(
                logits.reshape({-1, static_cast<int64_t>(tgt_vocab.size())}),
                labels.reshape({-1}),
                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(Vocab::PAD));
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();
            total_loss += loss.item<double>();
            ++batches;
        }
        std::cout << "Epoch " << epoch << "/" << cfg.epochs
                  << " - loss: " << (total_loss / std::max<std::size_t>(1, batches)) << '\n';
    }

    torch::save(model, (fs::path(model_dir) / "model.pt").string());
    json config = {
        {"src_vocab_size", src_vocab.size()}, {"tgt_vocab_size", tgt_vocab.size()},
        {"embedding_dim", cfg.embedding_dim}, {"hidden_dim", cfg.hidden_dim},
        {"attention_dim", cfg.attention_dim}, {"pad_id", Vocab::PAD},
        {"bos_id", Vocab::BOS}, {"eos_id", Vocab::EOS},
        {"max_length", cfg.max_length}
    };
    std::ofstream out(fs::path(model_dir) / "config.json");
    out << config.dump(2);
    std::cout << "Saved model and vocabularies to " << model_dir << '\n';
}

std::string translate_text(const std::string& model_dir, const std::string& text, int64_t max_length) {
    const fs::path dir(model_dir);
    auto src_vocab = Vocab::load((dir / "source_vocab.json").string());
    auto tgt_vocab = Vocab::load((dir / "target_vocab.json").string());
    std::ifstream in(dir / "config.json");
    if (!in) throw std::runtime_error("Cannot open model config.");
    json c; in >> c;

    Seq2Seq model(c.at("src_vocab_size").get<int64_t>(),
                  c.at("tgt_vocab_size").get<int64_t>(),
                  c.at("embedding_dim").get<int64_t>(),
                  c.at("hidden_dim").get<int64_t>(),
                  c.at("attention_dim").get<int64_t>(),
                  c.at("pad_id").get<int64_t>());
    torch::load(model, (dir / "model.pt").string());
    model->eval();

    auto ids = src_vocab.encode(text, true);
    auto source = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong)).unsqueeze(0);
    torch::NoGradGuard no_grad;
    auto prediction = model->translate(source, c.at("bos_id").get<int64_t>(),
        c.at("eos_id").get<int64_t>(), max_length);
    std::vector<int64_t> result;
    auto cpu = prediction.squeeze(0).to(torch::kCPU).contiguous();
    auto accessor = cpu.accessor<int64_t, 1>();
    for (int64_t i = 0; i < cpu.size(0); ++i) result.push_back(accessor[i]);
    return tgt_vocab.decode(result);
}
