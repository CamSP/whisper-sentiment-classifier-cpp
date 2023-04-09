#include <iostream>
#include <string>
#include <regex>
#include <fstream>
#include <map>
#include <vector>
#include <set>
#include <unordered_set>

using namespace std;

class Tokenizer {

    public:

    // Clase que tokeniza los textos 
    Tokenizer(){
        string stopword;
        string CorpusWord;
        // Carga de Stopwords
        ifstream stopword_file("../data/stopwords.txt");
        while (stopword_file >> stopword) {
            stopwords.insert(stopword);
        }
        
        //Carga de palabras aprendidas (Corpus)
        ifstream Corpus_file("../data/Corpus.txt");
        if (Corpus_file.is_open()) {
            while (Corpus_file >> CorpusWord) {
               Corpus.emplace(CorpusWord, Corpus.size());
            }
            Corpus_file.close();
        } else {
            // Si no existe el archivo Corpus, se crea
            ofstream new_file("../data/Corpus.txt");
            if (new_file.is_open()) {
                new_file.close();
            } else {
                cout << "Error: no se pudo crear el archivo Corpus.txt" << endl;
            }
        }
    }




    // Limpieza de palabras
    string preprocess(string str){
        // Se pasa el texto a minusculas
        transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return tolower(c); });
        // Patron que limpia:
        // Palabras de más de 13 caracteres
        // Números
        // URLS
        // Usuarios y hashtags de twitter
        // Signos de puntuación y signos especiales
        regex pattern("(\\w{13,})|(\\d+)|(https?:\\/\\/[^\\s]+)|(#[^\\s]+)|(@[^\\s]+)|(['´`]+)|([!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']+)");
        // Patron para eliminar espacios adicionales
        regex spaces("\\s{2,}");

        // Se limpia el texto
        str = regex_replace(str, pattern, "");
        
        // Eliminar stopwords
        stringstream ss(str);
        string word;
        string result;
        while (ss >> word) {
            if (stopwords.count(word) == 0) {
                result += word + ' ';
            }
        }

        // Se eliminan los espacios sobrantes
        str = regex_replace(result, spaces, " ");
        return str;
    }

    // Esta función se encarga de actualizar el Corpus con las palabras del string que entra
    void fit(string tempCorpus){
        // Patron que separa palabras
        regex delimiter("\\W+");
        // Limpieza del string
        string tempString = preprocess(tempCorpus);
        // iterador que separa las palabras
        sregex_token_iterator tempwords(tempString.begin(), tempString.end(), delimiter, -1);
        // Se añaden las palabras al corpus
        while (tempwords != reg_end) {
            Corpus.emplace(*tempwords++, Corpus.size());
        }

        // Se guarda el nuevo corpus en el archivo
        ofstream newCorpus("../data/Corpus.txt");
        if(newCorpus.is_open()){
            for(const auto& word: Corpus){
                newCorpus<<word<<"\n";
            }
        }
        newCorpus.close();
    }

    // Esta función se encarga de tokenizar 
    // vocab_size es el tamaño del vector de salida
    vector<int> tokenize(string str, size_t vocab_size){
        // Inicialización del vector
        vector<int> features(vocab_size, 0);
        // Patron que separa las palabras
        regex delimiter("\\W+");
        // string procesado
        string doc = preprocess(str);
        // iterador que separa las palabras
        sregex_token_iterator words(doc.begin(), doc.end(), delimiter, -1);
        
        for(int i = 0;words != reg_end && i < vocab_size;i++) {
            // Se busca la palabra en el corpus
            auto item = Corpus.find(*words++);
        
            if(item!=Corpus.end()){
                // Si existe, se almacena el el index de la palabra
                // Se suma 2 para reservar el 0 para valores sobrantes en el vector
                // y reservar el 1 para palabras desconocidas
                features[i] = item->second+2;
            }else{
                // Si se desconoce la palabra, se guarda un 1
                features[i] = 1;
            }
        }
        return features;
    }

    // Esta función combina las 2 anteriores
    // vocab_size es el tamaño del vector de salida
    vector<int> fit_tokenize(string str, size_t vocab_size){
        
        // Inicialización del vector
        vector<int> features(vocab_size, 0);
        // Patron que separa las palabras
        static const regex delimiter("\\W+");
        // string procesado
        string doc = preprocess(str);
        // iterador que separa las palabras
        sregex_token_iterator words(doc.begin(), doc.end(), delimiter, -1);
        
        for(int i = 0;words != reg_end && i < vocab_size;i++) {
            // Se añade la palabra en el corpus
            Corpus.emplace(*words, Corpus.size());
            // Se busca la palabra en el corpus
            auto item = Corpus.find(*words);
            if(item!=Corpus.end()){
                // Se almacena el index en el vector tokenizado
                features[i] = item->second+2;
            }
            *words++;
        }
        return features; 
    }

    // Guarda el nuevo corpus en el archivo Corpus.txt
    void saveFit(){
        ofstream newCorpus("../data/Corpus.txt");
        if(newCorpus.is_open()){
            for(const auto& word: Corpus){
                newCorpus<<word.first<<"\n";
            }
        }
        newCorpus.close();
    }


    private:
        unordered_set<string> stopwords;
        unordered_map<string, int> Corpus;
        sregex_token_iterator reg_end;
};