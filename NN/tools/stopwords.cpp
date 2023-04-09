#include <iostream>
#include <string>
#include <algorithm>
#include <regex>
#include <vector>
#include <fstream>
#include <map> 

using namespace std;

vector<string> palabras;

int main() {
    string s;
    ifstream file("../data/stopwords.txt");
    regex re("'");
    while(getline(file, s)){
        cout<<regex_replace(s, re, "")<<endl;
    }
}
