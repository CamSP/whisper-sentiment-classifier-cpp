#include <iostream>
#include <torch/torch.h>
#include "tokenizer.h"

class TData : public torch::data::datasets::Dataset<TData> {
    // Carga del dataset

    using Example = torch::data::Example<>;
    std::vector<std::vector<std::string>> content_dataset;
    std::vector<std::vector<int>> processed_dataset;

    public:
    // Constructor de la clase
    size_t vocab_size;
    // El parametro que recive define si se cargan los datos de
    // entrenamiento o de prueba
    TData(size_t vocab_size, bool isTestingData = false) {
        TData::vocab_size = vocab_size;
        if(isTestingData){
            content_dataset = testing_set();
            processed_dataset = preprocessDataset(false);
        }else{
            content_dataset = training_set();
            processed_dataset = preprocessDataset();
        }
        
    };

    // Carga del conjunto de datos de entrenamiento (Matriz)
    std::vector<std::vector<std::string>> training_set(){
        return read_dataset_csv("../data/train.csv");
    }

    // Carga del conjunto de datos de prueba (Matriz)
    std::vector<std::vector<std::string>> testing_set(){
        return read_dataset_csv("../data/test.csv");
    }


    // Preprocesamiento del dataset
    std::vector<std::vector<int>> preprocessDataset(bool train=true){
        // Matriz de salida
        std::vector<std::vector<int>>output_matrix;
        // Tokenizador
        Tokenizer tokenizer;
        // Inicialización de la matriz
        output_matrix.reserve(content_dataset.size());
        int i = 0;
        // Para cada columna del dataset
        for(const auto& row:content_dataset){    
            // Si es el set de entrenamiento, se entrena el corpus y se tokeniza
            // Si es el set de prueba, solo se tokeniza
            std::vector<int> tokenized = train?tokenizer.fit_tokenize(row[0], vocab_size):tokenizer.tokenize(row[0], vocab_size);
            // Se añade la categoria al final del dataset
            tokenized.push_back(std::stoi(row[1]));
            // Se añade el vector al dataset de salida
            output_matrix.push_back(tokenized);
            i++;
        }
        // Se guarda el nuevo corpus
        tokenizer.saveFit();
        return output_matrix;
    }

    // Separación entre datos y objetivos a predecir  
    Example get(size_t index){
        // Datos
        vector<int> tokens;
        // Cada elemento del vector tokenizado se añade al tensor de salida
        for(int i = 0; i<vocab_size;i++){
            tokens.push_back(processed_dataset[index][i]);
        }
        // Targets
        // Se añaden los objetivos al tensor de salida
        int sentiment = processed_dataset[index][vocab_size];
        // Convierte los datos en tensores
        return {torch::tensor(tokens), torch::tensor({sentiment})};
    }

    // Retorna el tamaño del dataset
    torch::optional<size_t> size() const {
        return processed_dataset.size();
    }

    // Lectura del csv
    // Retorna una matriz de strings
    std::vector<std::vector<std::string>> read_dataset_csv(std::string path){
        std::vector<std::vector<std::string>> content;
        std::vector<std::string> row;
        std::string line, word;
        
        
        std::fstream file (path, std::ios::in);
        // Se abre el archivo
        if(file.is_open()){
            while(getline(file, line)){
            row.clear();
            
            std::stringstream str(line);
            
            while(getline(str, word, '\t'))
                row.push_back(word);
                content.push_back(row);
            }

            // Cuando se acaba de leer el archivo, se muestra un mensaje de confirmación y el tamaño de los datos
            std::cout<<"Lectura completada, el tamaño es de: " << content.size() << std::endl;
        }
        else
        // Si hay un error en la carga, se muestra en consola
            std::cout<<"No se pudo abrir el archivo" << std::endl;

        return content;
    }
};