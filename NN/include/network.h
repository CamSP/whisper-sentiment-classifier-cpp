#pragma once

#include <torch/torch.h>
#include <iostream>

using torch::indexing::Slice;

// Arquitectura de la red neuronal
struct NetImpl : torch::nn::Module{
  // lstm -> Capa LSTM unidireccional
  // embedding -> Toma el vector de entrada y lo vectoriza
  // densa1 -> Capa fully connected, función de activación GeLu
  // out -> Capa de salida, función de activación Sigmoid
  // vocab_size = Tamaño del vocabulario
  // embedding_dim = Dimensiones de salida del embedding
  // hidden_size = Salida de la LSTM
  // layers = Número de capas de la LSTM
  public:
  NetImpl(size_t vocab_size, size_t embedding_dim, int hidden_size, size_t layers)
    : lstm(torch::nn::LSTMOptions(embedding_dim, hidden_size).num_layers(layers).batch_first(true).bidirectional(true).dropout(0.5)),
     densa1(hidden_size*2, 1),
     out(),
     embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim)){

      NetImpl::hidden_size = hidden_size;
      // Se registran los modulos en el modelo
      register_module("embedding", embedding);
      register_module("lstm", lstm);
      register_module("densa1", densa1);
      register_module("out", out);
    }

  // El metodo forward explica la sucesión de operaciones
  torch::Tensor forward(torch::Tensor x){
    // Al embedding le entra un tensor X
    x = embedding(x);
    // La salida del embedding entra a la LSTM
    auto lstm1 = std::get<0>(lstm(x));
    auto out_directions = lstm1.chunk(2, 2);
    auto out_1 = out_directions[0].index({Slice(), -1}); 
    auto out_2 = out_directions[1].index({Slice(), 0});
    auto out_cat = torch::cat({out_1, out_2}, 1);
    // La salida de la LSTM entra a la capa densa1
    auto densa = densa1(out_cat);
    // La salida de la capa densa entra a la neurona con activación sigmoide
    auto output = out(densa);
    // La salida es un numero entre 0 y 1
    return output;
  }

  torch::nn::Linear densa1{nullptr};
  torch::nn::Sigmoid out{nullptr};
  torch::nn::Embedding embedding{nullptr};
  torch::nn::LSTM lstm{nullptr};
  int hidden_size;
};

TORCH_MODULE(Net);


