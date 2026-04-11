// Comparison of the performances of serial and parallel
// Matrix Multiplication algorithm implementations

#include <omp.h>

#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// #include "ctrack.hpp"
#include "matrix.hpp"

// Serial algorithm
// Matrix A (dims mxn) * Matrix B (dims nxp) = Matrix C (dims mxp)
Matrix MM_ser(Matrix A, Matrix B) {
    // CTRACK;
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                int a = A.get_value_at(i, k);
                int b = B.get_value_at(k, j);
                int c = a * b;
                temp += c;
            }
            C.set_value_at(i, j, temp);
        }
    }
    return C;
}

// Simple parallel algorithm
Matrix MM_Par(Matrix A, Matrix B) {
    // CTRACK;
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    // TODO: complete

    return C;
}

// 1D Parallel algorithm
Matrix MM_1D(Matrix A, Matrix B, int p) {
    // CTRACK;
    int m = A.get_rows();
    int n = A.get_columns();
    int b_columns = B.get_columns();
    int number_of_rows_per_thread = A.get_rows() / p;

    Matrix C(m, b_columns);

    // TODO:

    return C;
}

// 2D Parallel algorithm
Matrix MM_2D(Matrix A, Matrix B, int p) {
    // CTRACK;
    int m1 = A.get_rows();
    int n1 = A.get_columns();
    int m2 = B.get_rows();
    int n2 = B.get_columns();

    Matrix C(m1, n2);

    // TODO:

    return C;
}

Matrix create_random_matrix(int rows, int columns, unsigned int seed = 5350) {
    std::mt19937 rng(seed);
    std::vector<int> v;

    int size = rows * columns;
    for (int i = 0; i < size; i++) {
        int r = (int)rng() % 1000;
        v.push_back(r);
    }

    Matrix C(rows, columns, v);

    return C;
}

int main(int argc, const char* argv[]) {
    // set up CLI args (makes it easier to run as a script)
    int m, n, q, P, seed;

    std::string fname;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "rows-A,m", po::value<int>(), "set amount of rows for matrix A")(
        "columns-A,n", po::value<int>(), "set amount of columns for matrix A")(
        "columns-B,q", po::value<int>(), "set amount of columns for matrix B")(
        "processors,P", po::value<int>(),
        "set number of processors for the parallel algorithms to use")(
        "output-file,o", po::value<std::string>()->default_value("results.txt"),
        "name of file containing ctrack output")(
        "seed,s", po::value<int>(), "set random seed for matrix generator");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << '\n';
    } else {
        if (vm.count("rows-A") && vm.count("columns-A") &&
            vm.count("columns-B") && vm.count("processors")) {
            m = vm["rows-A"].as<int>();
            n = vm["columns-A"].as<int>();
            q = vm["columns-B"].as<int>();
            P = vm["processors"].as<int>();

            if (vm.count("rows-A")) {
                fname = vm["output-file"].as<std::string>();
            }
            if (vm.count("seed")) {
                seed = vm["seed"].as<int>();
            }
        } else {
            std::cout << "Not all variables were set.\n";
            return -1;
        }
    }

    std::cout << "seed: " << seed << ", ";
    std::cout << "m: " << m << ", ";
    std::cout << "n: " << n << ", ";
    std::cout << "q: " << q << ", ";
    std::cout << "P: " << P << "\n";

    // actual code to run everything
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    Matrix a = create_random_matrix(m, n, seed);
    Matrix b = create_random_matrix(n, q, seed);
    Matrix c1 = MM_ser(a, b);
    Matrix c2 = MM_Par(a, b);
    Matrix c3 = MM_1D(a, b, P);
    Matrix c4 = MM_2D(a, b, P);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    // std::string results = ctrack::result_as_string();
    // std::ofstream out;
    // out.open(fname);
    // out << "--- SEED: " << seed << "---\n";
    // out << "m: " << m << ", ";
    // out << "n: " << n << ", ";
    // out << "q: " << q << ", ";
    // out << "P: " << P << "\n";
    // out << results;
    // out.close();

    auto duration = std::chrono::duration<double>(end - start);
    const auto hrs = std::chrono::duration_cast<std::chrono::hours>(duration);
    const auto mins =
        std::chrono::duration_cast<std::chrono::minutes>(duration - hrs);
    const auto secs =
        std::chrono::duration_cast<std::chrono::seconds>(duration - hrs - mins);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        duration - hrs - mins - secs);
    std::cout << "Duration: " << hrs.count() << "h" << mins.count() << "min"
              << secs.count() << "." << ms.count() << "sec" << std::endl;

    return 0;
}
