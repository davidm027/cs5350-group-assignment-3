# Create conda env
conda create -n my-env -c conda-forge gxx_linux-64=13 gcc_linux-64=13 cmake make pkg-config boost-cpp -y

# Activate conda env
conda activate my-env

# Build
chmod +x build.sh
./build.sh
