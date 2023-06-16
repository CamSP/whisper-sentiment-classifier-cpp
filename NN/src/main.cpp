#include "network.h"
#include "dataset.h"
#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <typeinfo>

using namespace torch;

// Parametros generales
struct Options {
    // data
    int input_dims = 100;

    // Embedding y LSTM
    int vocab_size = 114940;
    int embedding_dim = 64;
    int layers = 2;
    int hidden_size = 128;

    // Batch
    int train_batch_size = 64;
    int test_batch_size = 64;
    int epochs = 1000;
    float learning_rate = 0.001;

    torch::DeviceType device = (torch::cuda::is_available())?torch::kCUDA : torch::kCPU;
};

// Se crea el objeto con los parametros
static Options options;

// Función de entrenamiento
template <typename DataLoader>
// Recibe:
// network = Red neuronal
// loader = Datos de entrada
// optimizer = Optimizador 
// epoch = # de epocas de entrenamiento
// data_size = Tamaño de los datos

std::vector<float> train(
    Net& network,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t data_size) {
    // Se establece que la red neuronal va a ser entrenada
    network->train();

    //Se inicializa la perdida y la presición en 0
    float Loss = 0, Acc = 0;
    torch::nn::BCELoss bce_loss;
    // Se hace un loop en los datos donde se entrena con cada batch
    for (auto& batch : loader){
        // Se toman los datos
        auto data = batch.data.to(options.device);
        // Se toman los objetivos y se convierten en un tensor del tamaño del batch 
        auto targets = batch.target.view({options.train_batch_size, 1}).to(torch::kFloat).to(options.device);
        // Se evaluan los datos en la red neuronal
        auto output = network->forward(data);
        // Se evalua la perdida entre los datos predichos y los reales
        // (Cross Entropy es la función de perdida utilizada normalmente para salidas multiclase)
        auto loss = bce_loss(output, targets);
        // Dado que Sigmoid retorna un valor entre 0 y 1, se aproxima al entero más cercano
        // Se compara con los valores reales
        auto acc = torch::round(output).eq(targets).sum();
        // Se restaura el gradiente a cero
        optimizer.zero_grad();
        // Se evalua el gradiente para este batch
        loss.backward();
        // Se actualizan los parametros de la red neuronal
        optimizer.step();
        // Se añaden los datos de loss y acc a los valores historicos
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
        
    }
    // Despues de pasar por todos los datos, se imprimen los resultados de la epoca
    std::cout << "Train Loss: " << Loss/data_size << "\tAcc: " << Acc/data_size<< std::endl;
    std::vector<float> resume(2);
    resume[0] = Loss/data_size;
    resume[1] = Acc/data_size;
    return resume;
}


// Función de testing
template <typename DataLoader>
//Recibe:
// network = Red neuronal
// loader = Datos de entrada
// data_size = Tamaño de los datos

std::vector<float> test(Net& network, DataLoader& loader, size_t data_size) {
    // Se establece que la red neuronal va a ser probada (Se congelan sus parametros)
    network -> eval();

    // Inicialización de la perdida y la presición
    float Loss = 0, Acc = 0;
    torch::nn::BCELoss bce_loss;
    // Se hace un loop en los datos donde se entrena con cada batch
    for (const auto& batch : loader){
        // Se toman los datos
        auto data = batch.data.to(options.device);
        // Se toman los objetivos y se convierten en un tensor del tamaño del batch 
        auto targets = batch.target.view({options.test_batch_size, 1}).to(torch::kFloat).to(options.device);
        // Se evaluan los datos en la red neuronal
        auto output = network->forward(data);
        // Se calcula la loss
        auto loss = bce_loss(output, targets);
        // Se calcula la presición
        auto acc = torch::round(output);
        acc = acc.eq(targets).sum();
        // Se añaden los datos de loss y acc a los valores historicos
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    // Despues de pasar por todos los datos, se imprimen los resultados de la prueba
    std::cout << "Test  Loss: " << Loss/data_size<< "\tAcc: " << Acc/data_size << std::endl;
    std::vector<float> resume(2);
    resume[0] = Loss/data_size;
    resume[1] = Acc/data_size;
    return resume;
}

void save_results(std::string path, std::vector<std::vector<float>> values){
    std::ofstream file (path);
    for (auto& row : values) {
        int i = 0;
        for (auto col : row){
            file << col;
            if(i==0){
                file<<',';
            }
            i++;
        }
        file << '\n';
    }
    
}

int main(){
    // Cronometro
    auto start_time = std::chrono::high_resolution_clock::now();

    // Verificación de CUDA
    if(torch::cuda::is_available()){
        std::cout << "CUDA detectado, usando GPU" << std::endl;
    }else{
        std::cout << "Usando CPU" << std::endl;
    }
    
    // Carga del dataset de entrenamiento
    // La función map toma todos los tensores y los transforma en un unico tensor
    auto train_set = TData(options.input_dims).map(torch::data::transforms::Stack<>());

    // Se extrae el tamaño del dataset
    auto train_size = train_set.size().value();
    // Se crea el data loader del dataset
    // Esto divide los datos en batches 
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_set), options.train_batch_size);

    auto train_set_time = std::chrono::high_resolution_clock::now();
    auto train_set_duration = std::chrono::duration_cast<std::chrono::seconds>(train_set_time - start_time);
    std::cout<<"Dataset de entrenamiento procesado en: "<<train_set_duration.count()<<" segundos"<<std::endl;
    // Carga del dataset de prueba

    // La función map toma todos los tensores y los transforma en un unico tensor
    auto test_set = TData(options.input_dims, true).map(torch::data::transforms::Stack<>());
    
    // Se extrae el tamaño del dataset
    auto test_size = test_set.size().value();
    // Se crea el data loader del dataset
    // Esto divide los datos en batches 
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_set), options.test_batch_size);


    auto test_set_time = std::chrono::high_resolution_clock::now();
    auto test_set_duration = std::chrono::duration_cast<std::chrono::seconds>(test_set_time - train_set_time);
    std::cout<<"Dataset de prueba procesado en: "<<test_set_duration.count()<<" segundos"<<std::endl;
    // Creación de la red neuronal
    // 4 datos de entrada, 3 neuronas de salida
    Net network(options.vocab_size, options.embedding_dim, options.hidden_size, options.layers);
    // Se carga la red neuronal a la CPU
    network -> to(options.device);

    // Se crea el optimizador (Adam)
    // Recibe como parametro el learning rate
    torch::optim::Adam optimizer(
        network->parameters(), torch::optim::AdamOptions(options.learning_rate).weight_decay(0.001)
    );

    std::vector<std::vector<float>> resume_train;
    std::vector<std::vector<float>> resume_test;

    // Loop de entrenamiento por la cantidad de epocas seleccionadas
    for(size_t i = 0; i<options.epochs; ++i){
        auto epoch_time = std::chrono::high_resolution_clock::now();
        // Para cada epoca se entrena y se evalua
        std::cout<<"Epoch: "<<i+1<<std::endl;
        resume_train.push_back(train(network, *train_loader, optimizer, i+1, train_size));
        resume_test.push_back(test(network, *test_loader, test_size));
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_time);
        std::cout<<"Duración: "<<epoch_duration.count()<<" segundos"<<std::endl;
        std::cout << std::endl;
        if((i+1)%5==0){
            // Se guarda el historico cada 5 epocas
            torch::save(network, "../models/model_epoch_"+std::to_string(i+1)+".pt");
            save_results("../results/train_results.csv", resume_train);
            save_results("../results/test_results.csv", resume_test);
        }
    }

    // Se guarda el historico
    torch::save(network, "../models/model.pt");
    save_results("../results/train_results.csv", resume_train);
    save_results("../results/test_results.csv", resume_test);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout<<"El entrenamiento tardo: "<<duration.count()<<" segundos"<<std::endl;

    return 0;
};