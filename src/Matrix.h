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
    public:

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
    public:

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
};

template<typename T>
concept MX_MatrixType = requires (T t) {
    typename T::MX_matrix_type_tag; // Check if MX_matrix_type_tag exists
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
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = scalar_type<DstMatrixType>;
    using A_type = scalar_type<SrcMatrixType>;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
            }
        }
    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
            }
        }
    }

    return;
    
}

template<MX_MatrixType SrcMatrixType, MX_MatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = typename DstMatrixType::scalar_type;
    using A_type = typename SrcMatrixType::scalar_type;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, A.get_exp(i,j));
            }
        }
        
    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, A.get_exp(i,j));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, A.get_exp(i,j));
            }
        }
    }

    return;
    
}

template<MatrixType SrcMatrixType, MX_MatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = typename DstMatrixType::scalar_type;
    using A_type = typename SrcMatrixType::scalar_type;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, A.get_exp(i,j));
            }
        }
        
    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, A.get_exp(i,j));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, A.get_exp(i,j));
            }
        }
    }

    return;
    
}

template<MatrixType SrcMatrixType, MatrixType DstMatrixType>
using copy = lacpy<SrcMatrixType, DstMatrixType>;


template<MatrixType MatrixA, MatrixType MatrixA_t, int n_block_size = 1, int m_block_size = 1>
void transpose(MatrixA& A, MatrixA_t& At) {

    At.m = A.n;
    At.n = A.m;
    int n_blocks = (A.n + n_block_size - 1) / n_block_size;
    int m_blocks = (A.m + m_block_size - 1) / m_block_size;
    using At_type = typename MatrixA_t::scalar_type;
    using A_type = typename MatrixA::scalar_type;

    //break inrto cases - if layouts are different just copy over. If same, block the transpose
    if constexpr (MatrixA::layout == MatrixA_t::layout) {
        // blocked transpose
        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                for(int row = b_row*n_block_size; row < std::min(A.m, row + n_block_size); row++) {
                    for(int col = b_col*m_block_size; col < std::min(A.n, col + m_block_size); col++) {
                            At.data[At.get_idx(row, col)] = static_cast<At_type>(A.data[A.get_idx(col, row)]);
                    }
                }
            }
        }
    } else {
        // simnple copy
        for(int i = 0; i < A.rows()*A.cols(); i++) {
            At.data[i] = static_cast<At_type>(A.data[i]);
        }
    }

    
    
}


//for the case where At.r != A.r, just cast to a full precision matrix first and then cast back to a microsaled matrix
template<typename MX_MatrixA, typename MX_MatrixAt, int n_block_size, int m_block_size>
void transpose(MX_MatrixA& A, MX_MatrixAt& At)
{
    At.m = A.n;
    At.n = A.m;

    using A_type = MX_MatrixA::scalar_type;

    int n_blocks = (A.n + n_block_size - 1) / n_block_size;
    int m_blocks = (A.m + m_block_size - 1) / m_block_size;

    //cases - if layouts are different just copy over. If same, block the transpose
    if constexpr (MX_MatrixA::layout == MX_MatrixAt::layout) {
        // blocked transpose
        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                for(int row = b_row*n_block_size; row < std::min(A.m, row + n_block_size); row++) {
                    for(int col = b_col*m_block_size; col < std::min(A.n, col + m_block_size); col++) {
                            At.data[At.get_idx(row, col)] = static_cast<A_type>(A.data[A.get_idx(col, row)]);
                            if(A.get_idx(col, row) % A.r == 0) {
                                At.set_exp(row, col, A.get_exp(col, row));
                            }
                    }
                }
            }
        }
    } else {
        // simple copy
        for(int i = 0; i < A.rows()*A.cols(); i++) {
            At.data[i] = static_cast<A_type>(A.data[i]);
            At.set_exp(i, A.get_exp(i));
        }
    }



}

//slice to get submatrix
template<typename T, typename idx>
Matrix<T, idx> slice(const Matrix<T, idx>& a, range<idx> range) {
    assert(range.first >= 0 && range.second <= a.m);
    
    return Matrix<T, idx>(a.data + range.first * a.stride, range.second - range.first, a.n - range.first * a.stride);
}

template<typename T, typename idx, typename T_scal>
MX_Matrix<T, idx, T_scal> slice(const MX_Matrix<T, idx, T_scal>& a, range<idx> range) {
    assert(range.first >= 0 && range.second <= a.m);
    
    return MX_Matrix<T, idx, T_scal>(a.data + range.first * a.stride, a.shared_exps + range.first * a.r, range.second - range.first, a.n - range.first * a.r, a.stride);
}















} //namespace lo_float