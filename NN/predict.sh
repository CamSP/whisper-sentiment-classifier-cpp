clear
cd libtorch
path=$(pwd)
cd ..

mkdir results

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$path ..
make
./process > ../results/text_tagget.csv
