cd libtorch
path=$(pwd)
cd ..

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$path ..
cmake -DCMAKE_PREFIX_PATH=../../libtorch ..
make
time ./process > ../../results/text_tagget.csv
