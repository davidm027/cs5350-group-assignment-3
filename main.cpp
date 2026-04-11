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

#include "matrix.hpp"

// Serial algorithm
// Matrix A (dims mxn) * Matrix B (dims nxp) = Matrix C (dims mxp)
Matrix MM_ser(Matrix A, Matrix B) {
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
    int m = A.get_rows();
    int n = A.get_columns();
    int p = B.get_columns();

    Matrix C(m, p);

    // TODO: complete

    return C;
}

// 1D Parallel algorithm
Matrix MM_1D(Matrix A, Matrix B, int p) {
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

    // actual code to run everything
    Matrix a = create_random_matrix(m, n, seed);
    Matrix b = create_random_matrix(n, q, seed);

    auto start_ser = std::chrono::steady_clock::now();
    Matrix c1 = MM_ser(a, b);
    auto end_ser = std::chrono::steady_clock::now();
    auto duration_ser = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::duration<double>(end_ser - start_ser))
                            .count();

    auto start_par = std::chrono::steady_clock::now();
    Matrix c2 = MM_Par(a, b);
    auto end_par = std::chrono::steady_clock::now();
    auto duration_par = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::duration<double>(end_par - start_par))
                            .count();

    auto start_1d = std::chrono::steady_clock::now();
    Matrix c3 = MM_1D(a, b, P);
    auto end_1d = std::chrono::steady_clock::now();
    auto duration_1d = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::duration<double>(end_1d - start_1d))
                           .count();

    auto start_2d = std::chrono::steady_clock::now();
    Matrix c4 = MM_2D(a, b, P);
    auto end_2d = std::chrono::steady_clock::now();
    auto duration_2d = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::duration<double>(end_2d - start_2d))
                           .count();

    std::cout << m << ",";
    std::cout << n << ",";
    std::cout << q << ",";
    std::cout << P << ",";
    std::cout << duration_ser << ",";
    std::cout << duration_par << ",";
    std::cout << duration_1d << ",";
    std::cout << duration_2d << ",";
    std::cout << seed << "\n";

    return 0;
}