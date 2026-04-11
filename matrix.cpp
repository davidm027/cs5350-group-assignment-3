#include "matrix.hpp"
#include <iostream>
#include <sstream>

Matrix::Matrix(int m, int n) {
    this->rows = m;
    this->columns = n;
    int mxn = m * n;
    for (int i = 0; i < mxn; i++) {
        this->data.push_back(0);
    }
}

Matrix::Matrix(int m, int n, std::vector<int> data) {
    this->rows = m;
    this->columns = n;
    for (int i : data) {
        this->data.push_back(i);
    }
}

Matrix::~Matrix() {
    data.clear();
}

int Matrix::get_rows() {
    return this->rows;
}

int Matrix::get_columns() {
    return this->columns;
}

std::vector<int> Matrix::get_data() {
    return this->data;
}

int Matrix::get_value_at(int row, int col) {
    int idx = this->columns * row + col;
    return this->data[idx];
}

void Matrix::set_value_at(int row, int col, int val) {
    int idx = this->columns * row + col;
    this->data[idx] = val;
}

std::ostream& operator<<(std::ostream& os, Matrix m) {
    for (unsigned int i = 0; i < m.get_rows(); i++) {
        if (i == 0) {
            os << "[ ";
        } else {
            os << "  ";
        }
        for (unsigned int j = 0; j < m.get_columns(); j++) {
            os << m.get_value_at(i, j) << " ";
        }
        if (i == m.get_rows() - 1) {
            os << "]";
        }
        os << "\n";
    }
    return os;
}