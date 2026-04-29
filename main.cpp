// Comparison of the performances of serial and MPI 1D
// Matrix Multiplication algorithm implementations

#include <mpi.h>

#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// #include "ctrack.hpp"
#include "matrix.hpp"

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

Matrix transpose_matrix(Matrix B) {
    int rows = B.get_rows();
    int cols = B.get_columns();

    Matrix BT(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            BT.set_value_at(j, i, B.get_value_at(i, j));
        }
    }

    return BT;
}

std::vector<int> extract_row_block(Matrix M, int start_row, int num_rows) {
    int cols = M.get_columns();
    std::vector<int> block;
    block.reserve(num_rows * cols);

    for (int i = start_row; i < start_row + num_rows; i++) {
        for (int j = 0; j < cols; j++) {
            block.push_back(M.get_value_at(i, j));
        }
    }

    return block;
}

void place_row_block(Matrix& M,
                     int start_row,
                     const std::vector<int>& block,
                     int num_rows) {
    int cols = M.get_columns();
    int idx = 0;

    for (int i = start_row; i < start_row + num_rows; i++) {
        for (int j = 0; j < cols; j++) {
            M.set_value_at(i, j, block[idx++]);
        }
    }
}

void get_row_range(int total_rows,
                   int active_procs,
                   int rank,
                   int& start,
                   int& end) {
    if (rank >= active_procs) {
        start = 0;
        end = 0;
        return;
    }

    int rows_per_proc = total_rows / active_procs;
    start = rank * rows_per_proc;

    if (rank == active_procs - 1) {
        end = total_rows;
    } else {
        end = start + rows_per_proc;
    }
}

Matrix multiply_local_rows(Matrix A_local, Matrix BT) {
    int local_rows = A_local.get_rows();
    int n = A_local.get_columns();
    int q = BT.get_rows();

    Matrix local_C(local_rows, q);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < q; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += A_local.get_value_at(i, k) * BT.get_value_at(j, k);
            }
            local_C.set_value_at(i, j, temp);
        }
    }

    return local_C;
}

void multiply_add_local_blocks(const Matrix& A_block,
                               const Matrix& B_block,
                               Matrix& C_block) {
    int m = A_block.get_rows();
    int n = A_block.get_columns();
    int q = B_block.get_columns();

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            int a = A_block.get_value_at(i, k);
            for (int j = 0; j < q; j++) {
                int cur = C_block.get_value_at(i, j);
                cur += a * B_block.get_value_at(k, j);
                C_block.set_value_at(i, j, cur);
            }
        }
    }
}

bool matrices_equal(Matrix A, Matrix B) {
    if (A.get_rows() != B.get_rows() || A.get_columns() != B.get_columns()) {
        return false;
    }

    for (int i = 0; i < A.get_rows(); i++) {
        for (int j = 0; j < A.get_columns(); j++) {
            if (A.get_value_at(i, j) != B.get_value_at(i, j)) {
                return false;
            }
        }
    }

    return true;
}

// ------------------------------------------------------------
// Serial algorithm
// Matrix A (dims m x n) * Matrix B (dims n x q) = Matrix C (dims m x q)
// ------------------------------------------------------------

Matrix MM_ser(Matrix A, Matrix B) {
    // CTRACK;
    int m = A.get_rows();
    int n = A.get_columns();
    int q = B.get_columns();

    Matrix C(m, q);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += A.get_value_at(i, k) * B.get_value_at(k, j);
            }
            C.set_value_at(i, j, temp);
        }
    }

    return C;
}

// ------------------------------------------------------------
// MPI 1D algorithm
// Rank 0 owns full A and B initially
// Rank 0 transposes B into BT and sends:
//   - each worker its assigned rows of A
//   - the full BT
// Workers compute local rows of C and send them back to rank 0
// ------------------------------------------------------------

Matrix MM_1D_MPI(Matrix A, Matrix B, int rank, int world_size) {
    int m = 0;
    int n = 0;
    int q = 0;
    int active_procs = 0;

    if (rank == 0) {
        assert(A.get_columns() == B.get_rows());
        m = A.get_rows();
        n = A.get_columns();
        q = B.get_columns();
        active_procs = (m < world_size) ? m : world_size;
        if (active_procs <= 0) {
            active_procs = 1;
        }
    }

    if (rank == 0) {
        Matrix BT = transpose_matrix(B);
        std::vector<int> bt_data = BT.get_data();
        Matrix C(m, q);

        int meta[4];
        meta[0] = m;
        meta[1] = n;
        meta[2] = q;
        meta[3] = active_procs;

        for (int dest = 1; dest < world_size; dest++) {
            MPI_Send(meta, 4, MPI_INT, dest, 0, MPI_COMM_WORLD);

            int start, end;
            get_row_range(m, active_procs, dest, start, end);
            int local_rows = end - start;

            std::vector<int> a_block = extract_row_block(A, start, local_rows);

            MPI_Send(&local_rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);

            if (local_rows > 0) {
                MPI_Send(a_block.data(), local_rows * n, MPI_INT, dest, 2,
                         MPI_COMM_WORLD);
            }

            MPI_Send(bt_data.data(), q * n, MPI_INT, dest, 3, MPI_COMM_WORLD);
        }

        int root_start, root_end;
        get_row_range(m, active_procs, 0, root_start, root_end);
        int root_rows = root_end - root_start;

        std::vector<int> root_a_block =
            extract_row_block(A, root_start, root_rows);
        Matrix A_local(root_rows, n, root_a_block);
        Matrix local_C = multiply_local_rows(A_local, BT);

        std::vector<int> local_c_data = local_C.get_data();
        place_row_block(C, root_start, local_c_data, root_rows);

        for (int src = 1; src < world_size; src++) {
            int start, end;
            get_row_range(m, active_procs, src, start, end);
            int local_rows = end - start;

            std::vector<int> recv_block(local_rows * q);

            if (local_rows > 0) {
                MPI_Recv(recv_block.data(), local_rows * q, MPI_INT, src, 4,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                place_row_block(C, start, recv_block, local_rows);
            }
        }

        return C;
    } else {
        int meta[4];
        MPI_Recv(meta, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        m = meta[0];
        n = meta[1];
        q = meta[2];
        active_procs = meta[3];

        int local_rows = 0;
        MPI_Recv(&local_rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        std::vector<int> a_block(local_rows * n);
        if (local_rows > 0) {
            MPI_Recv(a_block.data(), local_rows * n, MPI_INT, 0, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::vector<int> bt_data(q * n);
        MPI_Recv(bt_data.data(), q * n, MPI_INT, 0, 3, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        Matrix A_local(local_rows, n, a_block);
        Matrix BT(q, n, bt_data);

        Matrix local_C = multiply_local_rows(A_local, BT);
        std::vector<int> local_c_data = local_C.get_data();

        if (local_rows > 0) {
            MPI_Send(local_c_data.data(), local_rows * q, MPI_INT, 0, 4,
                     MPI_COMM_WORLD);
        }

        return Matrix(0, 0);
    }
}

Matrix MM_2D(Matrix A, Matrix B, int rank, int world_size) {
    int m = 0;
    int n = 0;
    int q = 0;
    int active_procs = 0;
    int thread_dim = 0;

    if (rank == 0) {
        assert(A.get_columns() == B.get_rows());
        m = A.get_rows();
        n = A.get_columns();
        q = B.get_columns();

        thread_dim = (int)std::sqrt(world_size);

        assert(thread_dim > 0);
        assert(m % thread_dim == 0);
        assert(n % thread_dim == 0);
        assert(q % thread_dim == 0);

        assert(thread_dim * thread_dim == world_size);
        active_procs = thread_dim * thread_dim;

        int meta[5];
        meta[0] = m;
        meta[1] = n;
        meta[2] = q;
        meta[3] = thread_dim;
        meta[4] = active_procs;

        for (int dest = 1; dest < active_procs; dest++) {
            MPI_Send(meta, 5, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        int meta[5];
        MPI_Recv(meta, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        m = meta[0];
        n = meta[1];
        q = meta[2];
        thread_dim = meta[3];
        active_procs = meta[4];
    }

    int a_block_rows = m / thread_dim;
    int a_block_cols = n / thread_dim;
    int b_block_rows = n / thread_dim;
    int b_block_cols = q / thread_dim;
    int c_block_rows = m / thread_dim;
    int c_block_cols = q / thread_dim;
    int row = rank / thread_dim;
    int col = rank % thread_dim;

    Matrix C(m, q);

    Matrix A_local(a_block_rows, a_block_cols);
    Matrix B_local(b_block_rows, b_block_cols);
    Matrix C_local(c_block_rows, c_block_cols);

    if (rank == 0) {
        for (int dest = 1; dest < active_procs; dest++) {
            std::vector<int> vec_A;
            std::vector<int> vec_B;
            int p_row = dest / thread_dim;
            int p_col = dest % thread_dim;
            int a_row_start = p_row * a_block_rows;
            int a_col_start = p_col * a_block_cols;
            int b_row_start = p_row * b_block_rows;
            int b_col_start = p_col * b_block_cols;
            for (int i = 0; i < a_block_rows; i++) {
                for (int j = 0; j < a_block_cols; j++) {
                    vec_A.push_back(
                        A.get_value_at(a_row_start + i, a_col_start + j));
                }
            }
            for (int i = 0; i < b_block_rows; i++) {
                for (int j = 0; j < b_block_cols; j++) {
                    vec_B.push_back(
                        B.get_value_at(b_row_start + i, b_col_start + j));
                }
            }
            MPI_Send(vec_A.data(), a_block_rows * a_block_cols, MPI_INT, dest,
                     0, MPI_COMM_WORLD);
            MPI_Send(vec_B.data(), b_block_rows * b_block_cols, MPI_INT, dest,
                     1, MPI_COMM_WORLD);
        }

        int a_row_start = row * a_block_rows;
        int a_col_start = col * a_block_cols;

        int b_row_start = row * b_block_rows;
        int b_col_start = col * b_block_cols;

        std::vector<int> vec_A;
        for (int i = 0; i < a_block_rows; i++) {
            for (int j = 0; j < a_block_cols; j++) {
                vec_A.push_back(
                    A.get_value_at(a_row_start + i, a_col_start + j));
            }
        }
        std::vector<int> vec_B;
        for (int i = 0; i < b_block_rows; i++) {
            for (int j = 0; j < b_block_cols; j++) {
                vec_B.push_back(
                    B.get_value_at(b_row_start + i, b_col_start + j));
            }
        }

        for (int i = 0; i < a_block_rows; i++) {
            for (int j = 0; j < a_block_cols; j++) {
                A_local.set_value_at(i, j, vec_A.at(i * a_block_cols + j));
            }
        }
        for (int i = 0; i < b_block_rows; i++) {
            for (int j = 0; j < b_block_cols; j++) {
                B_local.set_value_at(i, j, vec_B.at(i * b_block_cols + j));
            }
        }
    } else {
        std::vector<int> tempA(a_block_rows * a_block_cols);
        MPI_Recv(tempA.data(), a_block_rows * a_block_cols, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> tempB(b_block_rows * b_block_cols);
        MPI_Recv(tempB.data(), b_block_rows * b_block_cols, MPI_INT, 0, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < a_block_rows; i++) {
            for (int j = 0; j < a_block_cols; j++) {
                A_local.set_value_at(i, j, tempA.at(i * a_block_cols + j));
            }
        }
        for (int i = 0; i < b_block_rows; i++) {
            for (int j = 0; j < b_block_cols; j++) {
                B_local.set_value_at(i, j, tempB.at(i * b_block_cols + j));
            }
        }
    }

    std::vector<int> sendA_buf(a_block_rows * a_block_cols);
    std::vector<int> recvA_buf(a_block_rows * a_block_cols);
    std::vector<int> sendB_buf(b_block_rows * b_block_cols);
    std::vector<int> recvB_buf(b_block_rows * b_block_cols);

    Matrix A_round(a_block_rows, a_block_cols);
    Matrix B_round(b_block_rows, b_block_cols);

    for (int round = 0; round < thread_dim; round++) {
        int round_block = round;
        int ownerA = row * thread_dim + round_block;
        int ownerB = round_block * thread_dim + col;

        if (col == round_block) {
            for (int i = 0; i < a_block_rows; i++) {
                for (int j = 0; j < a_block_cols; j++) {
                    A_round.set_value_at(i, j, A_local.get_value_at(i, j));
                }
            }

            int idx = 0;
            for (int i = 0; i < a_block_rows; i++) {
                for (int j = 0; j < a_block_cols; j++) {
                    sendA_buf[idx++] = A_local.get_value_at(i, j);
                }
            }

            for (int dest_col = 0; dest_col < thread_dim; dest_col++) {
                int dest = row * thread_dim + dest_col;
                if (dest != rank) {
                    MPI_Send(sendA_buf.data(), a_block_rows * a_block_cols,
                             MPI_INT, dest, 20 + round, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(recvA_buf.data(), a_block_rows * a_block_cols, MPI_INT,
                     ownerA, 20 + round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < a_block_rows; i++) {
                for (int j = 0; j < a_block_cols; j++) {
                    A_round.set_value_at(i, j,
                                         recvA_buf.at(i * a_block_cols + j));
                }
            }
        }
        if (row == round_block) {
            for (int i = 0; i < b_block_rows; i++) {
                for (int j = 0; j < b_block_cols; j++) {
                    B_round.set_value_at(i, j, B_local.get_value_at(i, j));
                }
            }

            int idx = 0;
            for (int i = 0; i < b_block_rows; i++) {
                for (int j = 0; j < b_block_cols; j++) {
                    sendB_buf[idx++] = B_local.get_value_at(i, j);
                }
            }

            for (int dest_row = 0; dest_row < thread_dim; dest_row++) {
                int dest = dest_row * thread_dim + col;
                if (dest != rank) {
                    MPI_Send(sendB_buf.data(), b_block_rows * b_block_cols,
                             MPI_INT, dest, 40 + round, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(recvB_buf.data(), b_block_rows * b_block_cols, MPI_INT,
                     ownerB, 40 + round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < b_block_rows; i++) {
                for (int j = 0; j < b_block_cols; j++) {
                    B_round.set_value_at(i, j,
                                         recvB_buf.at(i * b_block_cols + j));
                }
            }
        }
        multiply_add_local_blocks(A_round, B_round, C_local);
    }

    if (rank == 0) {
        for (int i = 0; i < c_block_rows; i++) {
            for (int j = 0; j < c_block_cols; j++) {
                C.set_value_at(i, j, C_local.get_value_at(i, j));
            }
        }

        for (int src = 1; src < active_procs; src++) {
            std::vector<int> recv_block(c_block_rows * c_block_cols);

            MPI_Recv(recv_block.data(), c_block_rows * c_block_cols, MPI_INT,
                     src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Matrix temp_C_local(c_block_rows, c_block_cols, recv_block);
            int p_row = src / thread_dim;
            int p_col = src % thread_dim;
            int c_row_start = p_row * c_block_rows;
            int c_col_start = p_col * c_block_cols;
            for (int i = 0; i < c_block_rows; i++) {
                for (int j = 0; j < c_block_cols; j++) {
                    C.set_value_at(c_row_start + i, c_col_start + j,
                                   temp_C_local.get_value_at(i, j));
                }
            }
        }
        return C;
    } else {
        std::vector<int> C_local_data = C_local.get_data();
        MPI_Send(C_local_data.data(), c_block_rows * c_block_cols, MPI_INT, 0,
                 4, MPI_COMM_WORLD);

        return Matrix(0, 0);
    }
}

Matrix create_random_matrix(int rows, int columns, unsigned int seed = 5350) {
    std::mt19937 rng(seed);
    std::vector<int> v;

    int size = rows * columns;
    for (int i = 0; i < size; i++) {
        int r = static_cast<int>(rng() % 1000);
        v.push_back(r);
    }

    Matrix C(rows, columns, v);
    return C;
}

int main(int argc, const char* argv[]) {
    MPI_Init(nullptr, nullptr);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int m = 0;
    int n = 0;
    int q = 0;
    int P = 0;
    int seed = 5350;
    std::string fname = "results.txt";

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
        if (rank == 0) {
            std::cout << desc << '\n';
        }
        MPI_Finalize();
        return 0;
    }

    if (!(vm.count("rows-A") && vm.count("columns-A") &&
          vm.count("columns-B") && vm.count("processors"))) {
        if (rank == 0) {
            std::cout << "Not all variables were set.\n";
        }
        MPI_Finalize();
        return -1;
    }

    m = vm["rows-A"].as<int>();
    n = vm["columns-A"].as<int>();
    q = vm["columns-B"].as<int>();
    P = vm["processors"].as<int>();

    if (vm.count("output-file")) {
        fname = vm["output-file"].as<std::string>();
    }
    if (vm.count("seed")) {
        seed = vm["seed"].as<int>();
    }

    Matrix a(0, 0);
    Matrix b(0, 0);

    if (rank == 0) {
        a = create_random_matrix(m, n, seed);
        b = create_random_matrix(n, q, seed);
    }

    Matrix c_serial(0, 0);
    Matrix c_mpi(0, 0);
    Matrix c_mpi_2d(0, 0);

    std::chrono::steady_clock::time_point serial_start;
    std::chrono::steady_clock::time_point serial_end;
    std::chrono::steady_clock::time_point mpi_start;
    std::chrono::steady_clock::time_point mpi_end;
    std::chrono::steady_clock::time_point mpi_2d_start;
    std::chrono::steady_clock::time_point mpi_2d_end;

    if (rank == 0) {
        serial_start = std::chrono::steady_clock::now();
        c_serial = MM_ser(a, b);
        serial_end = std::chrono::steady_clock::now();
    }

    mpi_start = std::chrono::steady_clock::now();
    c_mpi = MM_1D_MPI(a, b, rank, world_size);
    mpi_end = std::chrono::steady_clock::now();

    mpi_2d_start = std::chrono::steady_clock::now();
    c_mpi_2d = MM_2D(a, b, rank, world_size);
    mpi_2d_end = std::chrono::steady_clock::now();

    if (rank == 0) {
        bool correct = matrices_equal(c_serial, c_mpi);
        bool correct_2d = matrices_equal(c_serial, c_mpi_2d);

        std::cout << m << ",";
        std::cout << n << ",";
        std::cout << q << ",";
        std::cout << P << ",";

        auto serial_duration =
            std::chrono::duration<double>(serial_end - serial_start).count();
        auto mpi_duration =
            std::chrono::duration<double>(mpi_end - mpi_start).count();
        auto mpi_2d_duration =
            std::chrono::duration<double>(mpi_2d_end - mpi_2d_start).count();

        auto speedup_1d = (serial_duration / mpi_duration);
        auto cost_1d = (world_size * mpi_duration);
        auto speedup_2d = (serial_duration / mpi_2d_duration);
        auto cost_2d = (world_size * mpi_2d_duration);

        std::cout << serial_duration << ",";
        std::cout << mpi_duration << ",";
        std::cout << mpi_2d_duration << ",";
        std::cout << speedup_1d << ",";
        std::cout << cost_1d << ",";
        std::cout << speedup_2d << ",";
        std::cout << cost_2d << ",";
        std::cout << seed << "\n";
    }

    MPI_Finalize();
    return 0;
}
