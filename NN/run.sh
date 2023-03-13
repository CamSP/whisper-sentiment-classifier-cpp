cd libtorch
path=$(pwd)
cd ..

mkdir results
cd results
rm train_results.csv
rm test_results.csv
cd ..

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$path ..
make
./main

cd ..
mkdir graphs
cd tools
python3 trainResume.py