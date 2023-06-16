#include "network.h"
#include "dataset.h"
#include <iostream>
#include <torch/torch.h>

#include <typeinfo>

using namespace torch;
// Parametros generales
struct Options {
    int input_dims = 20;
    int vocab_size = 100000;
    int embedding_dim = 400;
    int hidden_size = 64;
    int layers = 2;

    torch::DeviceType device = torch::kCPU;
};

// Se crea el objeto con los parametros
static Options options;

int main(){
    Tokenizer tokenizer;
    Net network(options.vocab_size, options.embedding_dim, options.hidden_size, options.layers);
    torch::load(network, "../models/model.pt");
    network -> eval();

    std::string line;
    ifstream file("../whisperOutput/output.txt");
    while(getline(file, line)){
        vector<int>  texto_tokenizado = tokenizer.tokenize(line, 20);
        auto input_text = torch::from_blob(texto_tokenizado.data(), {1, 20}, torch::kInt32);
        torch::Tensor output = network->forward(input_text);
        output = torch::round(output);
        std::cout <<line<<"\t"<<output.data_ptr<float>()[0] << std::endl;
    }
    return 0;
}