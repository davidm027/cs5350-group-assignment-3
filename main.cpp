// Comparison of the performances of serial and parallel
// Matrix Multiplication algorithm implementations

#include <mpi.h>

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


// 1D Parallel algorithm

// issues to fix:
// I think we should check if matrix not square, basically last thread shoudl get whatever low left,
// not sure if indexing in transpose is correct need to fix!
// need to send calculation results back to process 0
Matrix MM_1D(Matrix A, Matrix B, int p) {

    if (p > A.get_rows())
        p = A.get_rows();
    int m = A.get_rows();
    int n = A.get_columns();
    int b_columns = B.get_columns();
    int number_of_rows_per_thread = A.get_rows() / p;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix B_T = B.transpose();
    Matrix C(m, b_columns);
    {
        int i, j, k;
        int thread_num = rank;
        int start = thread_num * number_of_rows_per_thread;
        int end = thread_num * number_of_rows_per_thread + number_of_rows_per_thread;
        if (thread_num == p - 1) {
            end = A.get_rows();
        }
        Matrix LocalA(number_of_rows_per_thread, n);
        Matrix LocalB_T(n, number_of_rows_per_thread);

        // sending data from process 0 to all the others (sending n/p chunks
        if (rank == 0) {
            for (int i = 1; i < p; i++) {
                MPI_Send(A.get_data().data() + i * number_of_rows_per_thread  *n , number_of_rows_per_thread * n, MPI_INT, i, 100,MPI_COMM_WORLD);
                MPI_Send(B_T.get_data().data() + i * number_of_rows_per_thread  *n , number_of_rows_per_thread * n, MPI_INT, i, 200,MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(LocalA.get_data().data() , number_of_rows_per_thread * n, MPI_INT, 0, 100,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(LocalB_T.get_data().data() , number_of_rows_per_thread * n, MPI_INT, 0, 200,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (i = start; i < end; i++) {
            for (j = 0; j < b_columns; j++) {
                int temp = 0;
                for (k = 0; k < n; k++) {
                    if (rank == 0) {
                        int a = A.get_value_at(i, k);
                        int b = B.get_value_at(k, j);
                        int c = a * b;
                        temp += c;
                    } else {
                        int a = LocalA.get_value_at(i - start, k);
                        int b = LocalB_T.get_value_at(j - start, k);
                        int c = a * b;
                        temp += c;
                    }
                }
                C.set_value_at(i, j, temp);
            }
        }
    }
    return C;
}

// 2D Parallel algorithm
Matrix MM_2D(Matrix A, Matrix B, int p) {
    // CTRACK;
    int m1 = A.get_rows();
    int n1 = A.get_columns();
    int n2 = B.get_columns();

    Matrix C(m1, n2);
    // int thread_dim = (int)std::sqrt(p);
    // int number_of_rows_per_thread = A.get_rows() / thread_dim;
    // int number_of_columns_per_thread = B.get_columns() / thread_dim;
    // omp_set_num_threads(p);
    //
    //
    // #pragma omp parallel shared(A, B, C)
    // {
    //     int i, j, k;
    //     int thread_num = omp_get_thread_num();
    //     int row = thread_num / thread_dim;
    //     int col = thread_num % thread_dim;
    //     int start = row * number_of_rows_per_thread;
    //     int end = m1;
    //     if (end <= m1) {
    //         end = start + number_of_rows_per_thread;
    //     }
    //
    //     int column_start = col * number_of_columns_per_thread;
    //
    //     int end_column = n2;
    //
    //     if (end_column <= n2) {
    //         end_column = column_start + number_of_columns_per_thread;
    //     }
    //
    //     int k_start = col * (n1/thread_dim);
    //
    //     int k_end = n1;
    //
    //     if (k_end <= n1) {
    //         k_end = k_start + (n1 / thread_dim);
    //
    //     }
    //
    //     for (i = start; i < end; i++) {
    //         for (j = 0; j < n2; j++) {
    //             int temp = 0;
    //             for (k = k_start; k < k_end; k++) {
    //                 int a = A.get_value_at(i, k);
    //                 int b = B.get_value_at(k, j);
    //                 temp += a * b;
    //             }
    //             #pragma omp critical
    //             {
    //                 temp += C.get_value_at(i, j);
    //                 C.set_value_at(i, j, temp);
    //             }
    //         }
    //     }
    // }
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

int main(int argc,  char* argv[]) {

    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);




    int m, n, q, P, seed;
    P = size;
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
    std::cout << duration_1d << ",";
    std::cout << duration_2d << ",";
    std::cout << seed << "\n";
    MPI_Finalize();

    return 0;
}