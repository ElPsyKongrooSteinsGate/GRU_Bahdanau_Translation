#include "model.h"
#include <vector>

EncoderImpl::EncoderImpl(int64_t vocab_size, int64_t embedding_dim, int64_t hidden_dim, int64_t pad_id)
    : hidden_dim_(hidden_dim), pad_id_(pad_id) {
    embedding = register_module("embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim).padding_idx(pad_id)));
    gru = register_module("gru",
        torch::nn::GRU(torch::nn::GRUOptions(embedding_dim, hidden_dim).batch_first(true)));
}

std::tuple<torch::Tensor, torch::Tensor> EncoderImpl::forward(torch::Tensor source) {
    const auto batch = source.size(0);
    const auto seq_len = source.size(1);
    auto options = embedding->weight.options();
    auto hidden = torch::zeros({1, batch, hidden_dim_}, options);
    std::vector<torch::Tensor> outputs;
    outputs.reserve(static_cast<std::size_t>(seq_len));

    // Step through tokens and freeze each sequence's hidden state after its PAD.
    // This avoids letting right-padding alter the encoder's final state.
    for (int64_t t = 0; t < seq_len; ++t) {
        auto token = source.select(1, t);
        auto embedded = embedding(token).unsqueeze(1);
        auto result = gru->forward(embedded, hidden);
        auto candidate_hidden = result.state;
        auto active = token.ne(pad_id_).view({1, batch, 1});
        hidden = torch::where(active, candidate_hidden, hidden);
        auto step_output = result.output.squeeze(1);
        step_output = step_output * active.squeeze(0).to(step_output.scalar_type());
        outputs.push_back(step_output);
    }
    return {torch::stack(outputs, 1), hidden};
}

BahdanauAttentionImpl::BahdanauAttentionImpl(int64_t hidden_dim, int64_t attention_dim) {
    encoder_projection = register_module("encoder_projection", torch::nn::Linear(hidden_dim, attention_dim));
    decoder_projection = register_module("decoder_projection", torch::nn::Linear(hidden_dim, attention_dim));
    energy_projection = register_module("energy_projection", torch::nn::Linear(attention_dim, 1));
}

std::pair<torch::Tensor, torch::Tensor> BahdanauAttentionImpl::forward(
    const torch::Tensor& encoder_outputs, const torch::Tensor& decoder_hidden,
    const torch::Tensor& source_mask) {
    auto query = decoder_hidden.squeeze(0).unsqueeze(1); // [B,1,H]
    auto energy = energy_projection->forward(torch::tanh(
        encoder_projection->forward(encoder_outputs) + decoder_projection->forward(query)
    )).squeeze(-1); // [B,S]
    energy = energy.masked_fill(source_mask.logical_not(), -1.0e9);
    auto weights = torch::softmax(energy, 1);
    auto context = torch::bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1); // [B,H]
    return {context, weights};
}

DecoderImpl::DecoderImpl(int64_t vocab_size, int64_t embedding_dim, int64_t hidden_dim,
                         int64_t attention_dim, int64_t pad_id) {
    embedding = register_module("embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim).padding_idx(pad_id)));
    attention = register_module("attention", BahdanauAttention(hidden_dim, attention_dim));
    gru = register_module("gru",
        torch::nn::GRU(torch::nn::GRUOptions(embedding_dim + hidden_dim, hidden_dim).batch_first(true)));
    output = register_module("output", torch::nn::Linear(hidden_dim + hidden_dim + embedding_dim, vocab_size));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DecoderImpl::forward(
    torch::Tensor input_token, torch::Tensor hidden,
    const torch::Tensor& encoder_outputs, const torch::Tensor& source_mask) {
    auto embedded = embedding(input_token).unsqueeze(1); // [B,1,E]
    auto attn = attention->forward(encoder_outputs, hidden, source_mask);
    auto context = attn.first;
    auto gru_input = torch::cat({embedded, context.unsqueeze(1)}, 2);
    auto result = gru->forward(gru_input, hidden);
    auto output_step = result.output.squeeze(1);
    auto new_hidden = result.state;
    auto logits = output->forward(torch::cat({output_step, context, embedded.squeeze(1)}, 1));
    return {logits, new_hidden, attn.second};
}

Seq2SeqImpl::Seq2SeqImpl(int64_t src_vocab, int64_t tgt_vocab, int64_t embedding_dim,
                         int64_t hidden_dim, int64_t attention_dim, int64_t pad_id)
    : pad_id_(pad_id), bos_id_(2), eos_id_(3) {
    encoder = register_module("encoder", Encoder(src_vocab, embedding_dim, hidden_dim, pad_id));
    decoder = register_module("decoder", Decoder(tgt_vocab, embedding_dim, hidden_dim, attention_dim, pad_id));
}

torch::Tensor Seq2SeqImpl::forward(torch::Tensor source, torch::Tensor target, double teacher_forcing_ratio) {
    auto enc = encoder->forward(source);
    auto encoder_outputs = std::get<0>(enc);
    auto hidden = std::get<1>(enc);
    auto source_mask = source.ne(pad_id_);
    auto input_token = target.select(1, 0); // BOS
    std::vector<torch::Tensor> logits_steps;
    const int64_t target_steps = target.size(1) - 1;
    for (int64_t t = 0; t < target_steps; ++t) {
        auto dec = decoder->forward(input_token, hidden, encoder_outputs, source_mask);
        auto logits = std::get<0>(dec);
        hidden = std::get<1>(dec);
        logits_steps.push_back(logits);
        if (t + 1 < target.size(1) && torch::rand({1}).item<double>() < teacher_forcing_ratio)
            input_token = target.select(1, t + 1);
        else
            input_token = logits.argmax(1);
    }
    return torch::stack(logits_steps, 1); // [B,T-1,V]
}

torch::Tensor Seq2SeqImpl::translate(torch::Tensor source, int64_t bos_id, int64_t eos_id, int64_t max_length) {
    auto enc = encoder->forward(source);
    auto encoder_outputs = std::get<0>(enc);
    auto hidden = std::get<1>(enc);
    auto source_mask = source.ne(pad_id_);
    auto input_token = torch::full({source.size(0)}, bos_id,
        torch::TensorOptions().dtype(torch::kLong).device(source.device()));
    std::vector<torch::Tensor> predictions;
    for (int64_t t = 0; t < max_length; ++t) {
        auto dec = decoder->forward(input_token, hidden, encoder_outputs, source_mask);
        hidden = std::get<1>(dec);
        input_token = std::get<0>(dec).argmax(1);
        predictions.push_back(input_token);
        if (input_token.eq(eos_id).all().item<bool>()) break;
    }
    if (predictions.empty())
        return torch::empty({source.size(0), 0}, source.options());
    return torch::stack(predictions, 1);
}
