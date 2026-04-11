# Create conda env
conda create -n my-env -c conda-forge gxx_linux-64=13 gcc_linux-64=13 cmake make pkg-config boost-cpp -y

# Activate conda env
conda activate my-env

# Build
mkdir -p build/

cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build/ --parallel