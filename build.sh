# Build
mkdir -p build/

cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build/ --parallel
