cd libtorch
path=$(pwd)
cd ..

mkdir results

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$path ..
cmake -DCMAKE_PREFIX_PATH=../../libtorch ..
make
./process > ../../results/text_tagget.csv
