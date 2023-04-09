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
            processed_dataset = preprocessDataset();
        }else{
            content_dataset = training_set();
            processed_dataset = preprocessDataset(false);
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

    std::vector<std::vector<int>> preprocessDataset(bool train=true){

        std::vector<std::vector<int>>output_matrix;
        Tokenizer tokenizer;
        output_matrix.reserve(content_dataset.size());
        int i = 0;
        for(const auto& row:content_dataset){    
            std::vector<int> tokenized = train?tokenizer.fit_tokenize(row[0], vocab_size):tokenizer.tokenize(row[0], vocab_size);
            tokenized.push_back(std::stoi(row[1]));
            output_matrix.push_back(tokenized);
            i++;

            if(i%10000==0)
                std::cout<<i<<std::endl;
        }

        tokenizer.saveFit();
        return output_matrix;
    }

    // Separación entre datos y objetivos a predecir  
    // stof pasa los datos de string a float
    Example get(size_t index){
        // Datos
        vector<int> tokens;
        for(int i = 0; i<vocab_size;i++){
            tokens.push_back(processed_dataset[index][i]);
        }
        // Targets
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