#include "network.h"
#include "dataset.h"

#include <iostream>
#include <torch/torch.h>

using namespace torch;

// Parametros generales
struct Options {
    int train_batch_size = 15;
    int test_batch_size = 45;
    int epochs = 20;
    float learning_rate = 0.01;
    torch::DeviceType device = torch::kCPU;
};

// Se crea el objeto con los parametros
static Options options;

// Función de entrenamiento
template <typename DataLoader>
//Recibe:
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
    // Index es el número de batch actual
    size_t index = 0;
    // Se establece que la red neuronal va a ser entrenada
    network->train();

    //Se inicializa la perdida y la presición en 0
    float Loss = 0, Acc = 0;

    // Se hace un loop en los datos donde se entrena con cada batch
    for (auto& batch : loader){
        // Se toman los datos
        auto data = batch.data.to(options.device);
        // Se toman los objetivos y se convierten en un tensor del tamaño del batch 
        auto targets = batch.target.to(options.device).view({options.train_batch_size, 3});
        // Se evaluan los datos en la red neuronal
        auto output = network->forward(data);
        // Se evalua la perdida entre los datos predichos y los reales
        // (Cross Entropy es la función de perdida utilizada normalmente para salidas multiclase)
        auto loss = torch::cross_entropy_loss(output, targets);
        // Se revisa que no hayan errores en el calculo de la perdida
        assert(!std::isnan(loss.template item<float>()));
        // Se evalua la predicción vs los datos reales
        // Dado que Adam retorna un arreglo de probabilidades, se toma el index del valor mas alto del arreglo
        // Se compara con el index más alto de los datos reales (One hot encoding)
        auto acc = output.argmax(1).eq(targets.argmax(1)).sum();

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
    std::cout << "Train Epoch: " << epoch << "\tLoss: " << Loss/data_size << "\tAcc: " << Acc/data_size<< std::endl;
    std::vector<float> resume(2);
    resume[0] = Loss/105;
    resume[1] = Acc/105;
    return resume;
}


// Función de testing
template <typename DataLoader>
//Recibe:
// network = Red neuronal
// loader = Datos de entrada
// data_size = Tamaño de los datos

std::vector<float> test(Net& network, DataLoader& loader, size_t data_size) {
    // Index es el número de batch actual
    size_t index = 0;
    // Se establece que la red neuronal va a ser probada (Se congelan sus parametros)
    network -> eval();
    torch::NoGradGuard no_grad;

    // Inicialización de la perdida y la presición
    float Loss = 0, Acc = 0;

    // Se hace un loop en los datos donde se entrena con cada batch
    for (const auto& batch : loader){
        // Se toman los datos
        auto data = batch.data.to(options.device);
        // Se toman los objetivos y se convierten en un tensor del tamaño del batch 
        auto targets = batch.target.to(options.device).view({options.test_batch_size, 3});
        // Se evaluan los datos en la red neuronal
        auto output = network->forward(data);
        // Se calcula la perdida
        auto loss = torch::cross_entropy_loss(output, targets);
        // Se revisa que no hayan errores en el calculo de la perdida
        assert(!std::isnan(loss.template item<float>()));
        // Se calcula la presición
        auto acc = output.argmax(1).eq(targets.argmax(1)).sum();

        // Se añaden los datos de loss y acc a los valores historicos
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    // Despues de pasar por todos los datos, se imprimen los resultados de la prueba
    std::cout << "Test Loss: " << Loss/options.test_batch_size<< "\tAcc: " << Acc/options.test_batch_size << std::endl;
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
    // Carga del dataset de entrenamiento
    // La función map toma todos los tensores y los transforma en un unico tensor
    auto train_set = TData().map(torch::data::transforms::Stack<>());
    // Se extrae el tamaño del dataset
    auto train_size = train_set.size().value();
    // Se crea el data loader del dataset
    // Esto divide los datos en batches 
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_set), options.train_batch_size);


    // Carga del dataset de prueba

    // La función map toma todos los tensores y los transforma en un unico tensor
    auto test_set = TData(true).map(torch::data::transforms::Stack<>());
    // Se extrae el tamaño del dataset
    auto test_size = test_set.size().value();
    // Se crea el data loader del dataset
    // Esto divide los datos en batches 
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_set), options.test_batch_size);

    // Creación de la red neuronal
    // 4 datos de entrada, 3 neuronas de salida
    Net network(4, 3);
    // Se carga la red neuronal a la CPU
    network -> to(options.device);

    // Se crea el optimizador (Adam)
    // Recibe como parametro el learning rate
    torch::optim::Adam optimizer(
        network->parameters(), torch::optim::AdamOptions(options.learning_rate)
    );

    std::vector<std::vector<float>> resume_train;
    std::vector<std::vector<float>> resume_test;

    // Loop de entrenamiento por la cantidad de epocas seleccionadas
    for(size_t i = 0; i<options.epochs; ++i){
        // Para cada epoca se entrena y se evalua
        resume_train.push_back(train(network, *train_loader, optimizer, i+1, train_size));
        resume_test.push_back(test(network, *test_loader, test_size));
        std::cout << std::endl;
    }

    // Se guarda el historico
    save_results("../results/train_results.csv", resume_train);
    save_results("../results/test_results.csv", resume_test);

    return 0;
}