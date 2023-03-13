#include <iostream>
#include <torch/torch.h>

class TData : public torch::data::datasets::Dataset<TData> {
    // Carga del dataset

    using Example = torch::data::Example<>;
    std::vector<std::vector<std::string>> content_dataset;

    public:
    // Constructor de la clase

    // El parametro que recive define si se cargan los datos de
    // entrenamiento o de prueba
    TData(bool isTestingData = false) {
        if(isTestingData){
            content_dataset = testing_set();
        }else{
            content_dataset = training_set();
        }
    };

    // Carga del conjunto de datos de entrenamiento (Matriz)
    std::vector<std::vector<std::string>> training_set(){
        return read_dataset_csv("../data/iris_train.csv");
    }

    // Carga del conjunto de datos de prueba (Matriz)
    std::vector<std::vector<std::string>> testing_set(){
        return read_dataset_csv("../data/iris_test.csv");
    }

    // Separación entre datos y objetivos a predecir  
    // stof pasa los datos de string a float
    Example get(size_t index){
        // Datos
        float sepal_length = stof(content_dataset[index][0]);
        float sepal_width = stof(content_dataset[index][1]);
        float petal_length = stof(content_dataset[index][2]);
        float petal_width = stof(content_dataset[index][3]);

        // Targets
        float specie1 = stof(content_dataset[index][4]);
        float specie2 = stof(content_dataset[index][5]);
        float specie3 = stof(content_dataset[index][6]);

        // Convierte los datos en tensores
        return {torch::tensor({sepal_length, sepal_width, petal_length, petal_width}), torch::tensor({specie1, specie2, specie3})};
    }

    // Retorna el tamaño del dataset
    torch::optional<size_t> size() const {
        return content_dataset.size();
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
            
            while(getline(str, word, ','))
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