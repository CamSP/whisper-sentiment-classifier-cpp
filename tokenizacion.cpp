#include <iostream>
#include <string>
#include <algorithm>
#include <regex>
#include <vector>
#include <fstream>
#include <map> 

using namespace std;

vector<string> palabras;
map<string, int> frecuencia;

int main() {
    string s;

    ifstream file("jtk.txt"); // Cambiar por el nombre del archivo que saca Whisper
    getline(file, s);

    regex re("\\W+");
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return tolower(c); });
    sregex_token_iterator it(s.begin(), s.end(), re, -1);
    sregex_token_iterator reg_end;
    while (it != reg_end) {
        palabras.push_back(*it++);
    }

    for (int i = 0; i < palabras.size(); i++) {
        frecuencia[palabras[i]]++;
    }

    for (auto it = frecuencia.begin(); it != frecuencia.end(); it++) {
        cout << it->first << " : " << it->second << endl;
    }
}
