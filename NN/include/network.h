#pragma once

#include <torch/torch.h>
#include <iostream>

// Arquitectura de la red neuronal
struct NetImpl : torch::nn::Module{
  // fc# -> Capas de la red neuronal
  // out -> Capa de salida (3 neuronas porque el dataset tiene 3 categorias)
  // input_dims = Dimensiones de la entrada de la red neuronal
  // output_dims = Dimensiones de la salida de la red neuronal
  public:
  NetImpl(int output_dims, size_t vocab_size, size_t embedding_dim, int hidden_size, size_t layers)
    : lstm(torch::nn::LSTMOptions(embedding_dim, hidden_size).num_layers(layers).dropout(0.5).batch_first(true)),
     densa1(hidden_size, 1),
     densa2(64, 32), 
     densa3(32, 1), 
     out(),
     embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim)){


      NetImpl::hidden_size = hidden_size;
      register_module("embedding", embedding);
      register_module("lstm", lstm);
      register_module("densa1", densa1);
      // register_module("densa2", densa2);
      // register_module("densa3", densa3);
      register_module("out", out);
    }

  // El metodo forward explica la sucesión de operaciones
  // En este caso, la función de activación para cada capa
  torch::Tensor forward(torch::Tensor x){
    //std::cout<<x<<std::endl;
    x = embedding(x);
    auto output = std::get<0>(lstm->forward(x)).index({torch::indexing::Slice(), -1});

    output = torch::gelu(densa1(output));
    // output = torch::tanh(densa2(output));
    // output = torch::tanh(densa3(output));
    // std::cout<<output<<std::endl;
    output = out(output);
    return output;
  }

  // Se establece el tipo de capa que es cada una las capas de la red
  // Linear es un capa fully connected comun y corriente
  torch::nn::Linear densa1{nullptr};
  torch::nn::Linear densa2{nullptr};
  torch::nn::Linear densa3{nullptr};
  torch::nn::Sigmoid out{nullptr};
  torch::nn::Embedding embedding{nullptr};
  torch::nn::LSTM lstm{nullptr};

  int hidden_size;
};

TORCH_MODULE(Net);


