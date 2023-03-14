#pragma once

#include <torch/torch.h>
#include <iostream>

// Arquitectura de la red neuronal
struct NetImpl : torch::nn::Module{
  // fc# -> Capas de la red neuronal
  // out -> Capa de salida (3 neuronas porque el dataset tiene 3 categorias)
  // input_dims = Dimensiones de la entrada de la red neuronal
  // output_dims = Dimensiones de la salida de la red neuronal
  NetImpl(int input_dims, int output_dims)
    : fc1(input_dims, 16), fc2(16, 32), out(32, output_dims){
      register_module("fc1", fc1);
      register_module("fc2", fc2);
      register_module("out", out);
    }

  // El metodo forward explica la sucesión de operaciones
  // En este caso, la función de activación para cada capa
  torch::Tensor forward(torch::Tensor x){
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = torch::nn::functional::softmax(out(x), torch::nn::functional::SoftmaxFuncOptions(1));
    return x;
  }

  // Se establece el tipo de capa que es cada una las capas de la red
  // Linear es un capa fully connected comun y corriente
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};
};

TORCH_MODULE(Net);


