#pragma once
#include "dataset.h"
#include "vocab.h"
#include "model.h"
#include <string>
#include <vector>

struct TrainConfig {
    int64_t embedding_dim = 128;
    int64_t hidden_dim = 256;
    int64_t attention_dim = 256;
    int64_t batch_size = 32;
    int64_t epochs = 20;
    double learning_rate = 0.001;
    double teacher_forcing_ratio = 0.5;
    int64_t max_length = 80;
};

void train_model(const std::string& db_path, const std::string& model_dir,
                 const TrainConfig& config);
std::string translate_text(const std::string& model_dir, const std::string& text,
                           int64_t max_length = 80);
