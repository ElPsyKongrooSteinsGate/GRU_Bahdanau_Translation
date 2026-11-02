#pragma once
#include <torch/torch.h>

struct EncoderImpl : torch::nn::Module {
    EncoderImpl(int64_t vocab_size, int64_t embedding_dim, int64_t hidden_dim, int64_t pad_id);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor source);
    torch::nn::Embedding embedding{nullptr};
    torch::nn::GRU gru{nullptr};
    int64_t hidden_dim_;
    int64_t pad_id_;
};
TORCH_MODULE(Encoder);

struct BahdanauAttentionImpl : torch::nn::Module {
    BahdanauAttentionImpl(int64_t hidden_dim, int64_t attention_dim);
    std::pair<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& encoder_outputs,
        const torch::Tensor& decoder_hidden,
        const torch::Tensor& source_mask);
    torch::nn::Linear encoder_projection{nullptr}, decoder_projection{nullptr}, energy_projection{nullptr};
};
TORCH_MODULE(BahdanauAttention);

struct DecoderImpl : torch::nn::Module {
    DecoderImpl(int64_t vocab_size, int64_t embedding_dim, int64_t hidden_dim,
                int64_t attention_dim, int64_t pad_id);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor input_token, torch::Tensor hidden,
        const torch::Tensor& encoder_outputs, const torch::Tensor& source_mask);
    torch::nn::Embedding embedding{nullptr};
    BahdanauAttention attention{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Linear output{nullptr};
};
TORCH_MODULE(Decoder);

struct Seq2SeqImpl : torch::nn::Module {
    Seq2SeqImpl(int64_t src_vocab, int64_t tgt_vocab, int64_t embedding_dim,
                int64_t hidden_dim, int64_t attention_dim, int64_t pad_id);
    torch::Tensor forward(torch::Tensor source, torch::Tensor target,
                          double teacher_forcing_ratio);
    torch::Tensor translate(torch::Tensor source, int64_t bos_id, int64_t eos_id,
                            int64_t max_length);
    Encoder encoder{nullptr};
    Decoder decoder{nullptr};
    int64_t pad_id_;
    int64_t bos_id_;
    int64_t eos_id_;
};
TORCH_MODULE(Seq2Seq);
