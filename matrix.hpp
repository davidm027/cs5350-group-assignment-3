#include <ostream>
#include <string>
#include <vector>

class Matrix {
   private:
    int rows;
    int columns;
    std::vector<int> data;

   public:
    Matrix(int, int);
    Matrix(int, int, std::vector<int>);
    ~Matrix();

    int get_rows() const;
    int get_columns() const;
    const std::vector<int>& get_data() const;
    int get_value_at(int, int) const;
    void set_value_at(int, int, int);

    friend std::ostream& operator<<(std::ostream&, const Matrix&);
};