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
    size_t number_of_features;

    Tokenizer(size_t n_features=50){
        number_of_features = n_features;
        string stopword;
        string CorpusWord;
        // Carga de Stopwords
        ifstream stopword_file("../data/stopwordsprocessed.txt");
        while (stopword_file >> stopword) {
            stopwords.insert(stopword);
        }
        
        //Carga de palabras aprendidas
        ifstream Corpus_file("../data/Corpus.txt");
        if (Corpus_file.is_open()) {
            while (Corpus_file >> CorpusWord) {
               Corpus.emplace(CorpusWord, Corpus.size());
            }
            Corpus_file.close();
        } else {
            ofstream new_file("../data/Corpus.txt");
            if (new_file.is_open()) {
                new_file.close();
            } else {
                cout << "Error: no se pudo crear el archivo Corpus.txt" << endl;
            }
        }
    }




    string preprocess(string str){
        // Limpieza de palabras
        transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return tolower(c); });

        regex pattern("(\\w{13,})|(\\d+)|(https?:\\/\\/[^\\s]+)|(#[^\\s]+)|(@[^\\s]+)|(['´`]+)|([!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']+)");
        regex spaces("\\s+");
        str = regex_replace(str, pattern, "");
        
        // Eliminar palabras en stopwords
        stringstream ss(str);
        string word;
        string result;
        while (ss >> word) {
            if (stopwords.count(word) == 0) {
                result += word + ' ';
            }
        }
        str = regex_replace(result, spaces, " ");
        return str;
    }

    void fit(string tempCorpus){

        regex delimiter("\\W+");
        string tempString = preprocess(tempCorpus);
        sregex_token_iterator tempwords(tempString.begin(), tempString.end(), delimiter, -1);
        while (tempwords != reg_end) {
            Corpus.emplace(*tempwords++, Corpus.size());
        }


        ofstream newCorpus("../data/Corpus.txt");
        if(newCorpus.is_open()){
            for(const auto& word: Corpus){
                newCorpus<<word<<"\n";
            }
        }
        newCorpus.close();
    }

    vector<int> tokenize(string str, size_t vocab_size){
        
        vector<int> features(vocab_size, 0);
        regex delimiter("\\W+");
        
        string doc = preprocess(str);
        
        sregex_token_iterator words(doc.begin(), doc.end(), delimiter, -1);
        
        for(int i = 0;words != reg_end && i < vocab_size;i++) {
            auto item = Corpus.find(*words++);
            if(item!=Corpus.end()){
                features[i] = item->second+2;
            }else{
                features[i] = 1;
            }
        }
        
        return features;
    }

    vector<int> fit_tokenize(string str, size_t vocab_size){
        
        vector<int> features(vocab_size, 0);
        static const regex delimiter("\\W+");
        
        string doc = preprocess(str);
        
        sregex_token_iterator words(doc.begin(), doc.end(), delimiter, -1);
        
        for(int i = 0;words != reg_end && i < vocab_size;i++) {
            Corpus.emplace(*words, Corpus.size());
            auto item = Corpus.find(*words);
            if(item!=Corpus.end()){
                features[i] = item->second+2;
            }
            *words++;
        }
        return features; 
    }


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