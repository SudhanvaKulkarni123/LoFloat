/// @author Sudhanva Kulkarni
/// Simplke implementation of Varying Layout dense matrix

#include <concepts>
#include <cassert>
#include "layouts.h"


using namespace std;

namespace lo_float {

template<typename T>
using range = std::pair<T, T>;

template<typename T, typename idx, Layout L = ColMajor>
class Matrix {
    
    idx m;
    idx n;
    idx ld;
    T* data;
    static constexpr Layout layout = L;
    using scalar_type = T;
    using matrix_type_tag = void;

    
    Matrix(T* data, idx m, idx n, idx ld) : data(data), m(m), n(n), ld(ld) {}

    constexpr inline T& operator() const (idx row, idx col) {
       return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((*this)(i,j))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((*this)(i,j))) return true;
            }
        }
        return false;
    }

    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }

};

template<typename T, typename idx, typename T_scal , Layout L = ColMajor>
class MX_Matrix {
    
    idx m;
    idx n;
    idx ld;
    idx r;  //numbver  of floats that share an exp
    T* data;
    T_scal* shared_exps;
    static constexpr Layout layout = L;
    using scalar_type = T;
    using shared_exp_type = T_scal;
    using MX_matrix_type_tag = void;

    
    MX_Matrix(T* data, T_scal* shared_exp, idx m, idx n, idx ld, idx r) : data(data), shared_exp(shared_exp), m(m), n(n), ld(ld), r(r) {}

    constexpr inline idx get_idx(idx row, idx col) const {
        if constexpr (L == ColMajor) return col*ld + row;
        else return row*ld + col;
    }

    constexpr inline T& operator() const (idx row, idx col) {
        return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    } 

    constexpr inline T_scal get_exp(idx row, idx col) const {
        if constexpr (L == Layout::ColMajor) {
            return T_scal[(col*ld + row)/r];
        } else {
            return T_scal[(row*ld + col)/r];
        }
    }

    template<typename T>
    constexpr inline void set_exp(idx row, idx col, T value) const {

        if constexpr (L == Layout::ColMajor) {
            T_scal[(col*ld + row)/r] = std::is_integral_v<shared_exp_type> ? static_cast<shared_exp_type>(log2(value)) : static_cast<shared_exp_type>(value);
        } else {
            T_scal[(row*ld + col)/r] = std::is_integral_v<shared_exp_type> ? static_cast<shared_exp_type>(log2(value)) : static_cast<shared_exp_type>(value);
        }
    }

    template<typename V>
    constexpr inline V scaled_val(idx row, idx col) const {
        return operator()(row, col) * get_exp(row, col);
    }


    constexpr inline T& operator() const (idx row, idx col) {
        return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((this)->scaled_val(i,j))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((this)->scaled_val(i,j))) return true;
            }
        }
        return false;
    }


    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }



};

// Concept: matches if T has either matrix_type_tag or MX_matrix_type_tag
template<typename T>
concept MatrixType = requires (T t) {
    typename T::matrix_type_tag;    // Check if matrix_type_tag exists
} || requires (T t) {
    typename T::MX_matrix_type_tag; // OR check if MX_matrix_type_tag exists
};

// Trait: true if T has MX_matrix_type_tag
template<typename T>
struct is_MX_MatrixType {
    static constexpr bool value = requires { typename T::MX_matrix_type_tag; };
};

template<typename T>
using is_MX_MatrixType_t = typename is_MX_MatrixType<T>::value;

// Concept: matches only if T has matrix_type_tag (Vanilla Matrix)
template<typename T>
struct is_Vanilla_MatrixType = requires {
    typename T::matrix_type_tag;
};

template<typename T>    
struct is_Vanilla_MatrixType_t {
    static constexpr bool value = requires { typename T::matrix_type_tag; };
};

//helper that returns if the format is MX by checking if the type has a shared_exps member
template<typename T>
struct is_MX_format {
    static constexpr bool value = false;
};
template<typename T, typename idx, typename T_scal, Layout L, MX_Layout MX_L>
struct is_MX_format<MX_Matrix<T, idx, T_scal, L, MX_L>> {
    static constexpr bool value = true;
};
template<typename T>
struct is_MX_format<Matrix<T, idx>> {
    static constexpr bool value = false;
};


template<MatrixType SrcMatrixType, MatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = scalar_type<DstMatrixType>;
    //if both are vanilla matrices, just copy over data
    
    if constexpr (is_Vanilla_MatrixType<SrcMatrixType>::value && is_Vanilla_MatrixType<DstMatrixType>::value) {
        std::memcpy(B.data, A.data, A.rows()*A.cols()*sizeof(typename SrcMatrixType::scalar_type));
    } else if constexpr (is_MX_MatrixType<SrcMatrixType>::value && is_MX_MatrixType<DstMatrixType>::value) {
        //if both are MX matrices, check if r values are same. If yes, make two calls to memcpy. If they are diferent, then create a small buffer to use an intermediate
        if constexpr (A.r == B.r) {
            std::memcpy(B.data, A.data, A.rows()*A.cols()*sizeof(typename SrcMatrixType::scalar_type));
            std::memcpy(B.shared_exp, A.shared_exp, A.rows()*A.cols()*sizeof(typename SrcMatrixType::shared_exp_type)/SrcMatrixType::r);
        } else {
            //create a small buffer to use as intermediate
            int r1 = A.r;
            int r2 = B.r;
            float* buffer = new float[r2];
            //for 
            if(A::layout == Layout::RowMajor) {
                for(int i = 0; i < A.rows(); i++) {
                    for(int j_block = 0; j_block < A.cols()/A.r; j_block++) {
                        //get true values of A
                        float blk_max = 0.0f;
                        for(int j = 0; j < B.r; j++) {
                            buffer[j] = A.scaled_val(i, j_block*A.r + j);
                            blk_max = std::max(blk_max, abs(buffer[j]));
                        }
                        for(int j = 0; j < B.r; j++) {
                            B(i, j_block*A.r +j) = static_cast<B_type>(buffer[j]/blk_max);
                            buffer[j] = 0.0f;
                        }
                        blk_max = 0.0f;

                    }
                }
            } else {
                for(int i = 0; i < A.cols(); i++) {
                    for(int j_block = 0; j_block < A.rows()/A.r; j_block++) {
                        //get true values of A
                        float blk_max = 0.0f;
                        for(int j = 0; j < B.r; j++) {
                            buffer[j] = A.scaled_val(i, j_block*A.r + j);
                            blk_max = std::max(blk_max, abs(buffer[j]));
                        }
                        for(int j = 0; j < B.r; j++) {
                            B(i, j_block*A.r +j) = static_cast<B_type>(buffer[j]/blk_max);
                            buffer[j] = 0.0f;
                        }
                        blk_max = 0.0f;

                    }
                }
            }
            

            
            
        }
        
    } else if constexpr (is_Vanilla_MatrixType<SrcMatrixType>::value && is_MX_MatrixType<DstMatrixType>::value) {

        if constexpr (SrcMatrixType::layout == Layout::ColMajor) {
            for(int i = 0; i < A.rows(); i++) {
                for(int j = 0; j < A.cols()/A.r; j++) {
                    //TODO : parallelize with OpenMP
                    auto blk_max = static_cast<typename SrcMatrixType::scalar_type>(0.0);
                    for(int k = 0; k < A.r; k++) {
                        blk_max = std::max(blk_max, A(i,j*A.r + k));
                    }
                    B.set_exp(i, j*A.r, blk_max);
                    
                   
                }
            }
        } else {
            for(int i = 0; i < A.rows(); i++) {
                for(int j = 0; j < A.cols(); j++) {
                    B.data[B.get_idx(i,j)] = A.data[A.get_idx(j,i)];
                }
            }
        }
       
    }
}


template<typename MatrixA, typename MatrixA_t, int n_block_size, int m_block_size>
void transpose(MatrixA& A, MatrixA_t& At) {

    At.m = A.n;
    At.n = A.m;
    int n_blocks = (A.n + n_block_size - 1) / n_block_size;
    int m_blocks = (A.m + m_block_size - 1) / m_block_size;
    if constexpr (At::layout != A::layout) {
        //just copy over the data with memcpy
        std::memcpy(At.data, A.data, A.m*A.n*sizeof(typename MatrixA::value_type));
    } else {
        // blocked transpose
        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                for(int row = b_row*n_block_size; row < std::min(A.m, row + n_block_size); row++) {
                    for(int col = b_col*m_block_size; col < std::min(A.n, col + m_block_size); col++) {
                            At.data[At.get_idx(row, col)] = A.data[A.get_idx(col, row)];
                    }
                }
            }
        }
    }
    
}


//main edge case here is when A.r != At.r
template<typename MX_MatrixA, typename MX_MatrixAt, int n_block_size, int m_block_size>
void transpose(MX_MatrixA& A, MX_MatrixAt& At)
{
    At.m = A.n;
    At.n = A.m;

    using A_type = MX_MatrixA::scalar_type;

    int n_blocks = (A.n + n_block_size - 1) / n_block_size;
    int m_blocks = (A.m + m_block_size - 1) / m_block_size;

    if constexpr (A::layout != At::layout) {
        std::memcpy(At.data, A.data, A.m*A.n*sizeof(typename MatrixA::value_type));
        if constexpr (A.r = At.r) {
            std::memcpy(A.shared_exp, At.shared_exp, A.m*A.n*sizeof(typename MatrixA::value_type)/A.r);

        } else {
            //traverse At and set exponents
            if constexpr (At::layout == Layout::ColMajor) {
                for(int i = 0; i < At.n; i++) {
                    //TODO : parallelize with OpenMP
                    for(int j_block = 0; j_block < At.m/r; j_block++) {
                        //find max
                        auto blk_max = static_cast<A_type>(0.0);
                        for(int j = j_block*r; j < std::min(At.m, j + r); j++) {
                            blk_max = std::max(blk_max, At(i,j));
                        }
                        //set blk_max to exp array -
                        At.set_exp(i, j_block*r, blk_max);
                    }
                }
            } else {
                for(int i = 0; i < At.m; i++) {
                    //TODO : parallelize with OpenMP
                    for(int j_block = 0; j_block < At.n/r; j_block++) {
                        //find max
                        auto blk_max = static_cast<A_type>(0.0);
                        for(int j = j_block*r; j < std::min(At.n, j + r); j++) {
                            blk_max = std::max(blk_max, At(i,j));
                        }
                        //set blk_max to exp array -
                        At.set_exp(i, j_block*r, blk_max);

                    }
                }
            }
        }
    } else {
        // blocked transpose
        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                for(int row = b_row*n_block_size; row < std::min(A.m, row + n_block_size); row++) {
                    for(int col = b_col*m_block_size; col < std::min(A.n, col + m_block_size); col++) {
                            At.data[At.get_idx(row, col)] = A.data[A.get_idx(col, row)];
                    }
                }
            }
        }

    }


}


/// @brief @brief
/// @tparam packing_type 
/// @tparam Matrix_t 
/// @param A 
/// @param rows 
/// @param cols 
/// @param packing_buff 
/// @param trans 
template<typename Matrix_t, typename packing_type>
void pack_submatrix(Matrix_t& A, range<int> rows_range, range<int> cols_range, packing_type* packing_buff, bool trans)
{
    int row_start = rows_range.first;
    int row_end   = rows_range.second;
    int col_start = cols_range.first;
    int col_end   = cols_range.second;

    int rows = row_end - row_start;
    int cols = col_end - col_start;

    if (trans) {
        // Pack with transpose: (i,j) -> (j,i)
        if constexpr (Matrix_t::layout == Layout::ColMajor) {
            for (int j = row_start; j < row_end; ++j) {
                for (int i = col_start; i < col_end; ++i) {
                    packing_buff[(j - col_start) * cols + (i - row_start)] = A(i, j);
                }
            }
        } else {
        for (int j = col_start; j < col_end; ++j) {
            for (int i = row_start; i < row_end; ++i) {
                packing_buff[(j - col_start) * rows + (i - row_start)] = A(i, j);
            }
        }
    }
    } else {
        if constexpr (Matrix_t::layout == Layout::ColMajor) {
            for (int j = col_start; j < col_end; ++j) {
                for (int i = row_start; i < row_end; ++i) {
                    packing_buff[(i - row_start) + (j - col_start) * rows] = A(i, j);
                }
            }
        } else { // RowMajor
            for (int i = row_start; i < row_end; ++i) {
                for (int j = col_start; j < col_end; ++j) {
                    packing_buff[(j - col_start) + (i - row_start) * cols] = A(i, j);
                }
            }
        }
    }
}

template<typename Matrix_t, typename packing_type>
void unpack_submatrix(Matrix_t& A, range<int> rows_range, range<int> cols_range, packing_type* packing_buff, bool trans)
{
    int row_start = rows_range.first;
    int row_end   = rows_range.second;
    int col_start = cols_range.first;
    int col_end   = cols_range.second;

    int rows = row_end - row_start;
    int cols = col_end - col_start;

    if (trans) {
        // Pack with transpose: (i,j) -> (j,i)
        if constexpr (Matrix_t::layout == Layout::ColMajor) {
            for (int j = row_start; j < row_end; ++j) {
                for (int i = col_start; i < col_end; ++i) {
                    A(i,j) = packing_buff[(j - col_start) * cols + (i - row_start)];
                }
            }
        } else {
        for (int j = col_start; j < col_end; ++j) {
            for (int i = row_start; i < row_end; ++i) {
                A(i,j) = packing_buff[(j - col_start) * rows + (i - row_start)];
            }
        }
    }
    } else {
        if constexpr (Matrix_t::layout == Layout::ColMajor) {
            for (int j = col_start; j < col_end; ++j) {
                for (int i = row_start; i < row_end; ++i) {
                    A(i,j) = packing_buff[(i - row_start) + (j - col_start) * rows];
                }
            }
        } else { // RowMajor
            for (int i = row_start; i < row_end; ++i) {
                for (int j = col_start; j < col_end; ++j) {
                    A(i,j) = packing_buff[(j - col_start) + (i - row_start) * cols];
                }
            }
        }
    }

}













} //namespace lo_float