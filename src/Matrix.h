/// @author Sudhanva Kulkarni
/// Simplke implementation of Varying Layout dense matrix
#include <algorithm>
#include <concepts>
#include <type_traits>
#include <cassert>
#include "lo_float.h"
#include "layouts.h"



using namespace std;
using namespace lo_float;

namespace Lo_Gemm {



template<Float T, Int idx = int, Layout L = ColMajor>
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

    inline T& operator()(idx row, idx col) const {
        if constexpr (L == ColMajor)
            return data[col * ld + row];
        else
            return data[row * ld + col];
    }
    
    // Added for consistency and use in transpose function
    constexpr inline idx get_idx(idx row, idx col) const {
        if constexpr (L == ColMajor)
            return col * ld + row;
        else
            return row * ld + col;
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((*this)(row,col))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((*this)(row,col))) return true;
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

template<Float T, Float T_scal , Int idx = int,Layout L = ColMajor>
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

    
    MX_Matrix(T* data, T_scal* shared_exps, idx m, idx n, idx ld, idx r) : data(data), shared_exps(shared_exps), m(m), n(n), ld(ld), r(r) {}

    constexpr inline idx get_idx(idx row, idx col) const {
        if constexpr (L == ColMajor) return col*ld + row;
        else return row*ld + col;
    }

    inline T& operator() (idx row, idx col) const {
        if constexpr (L == ColMajor)
            return data[col*ld + row];
        else
            return data[row*ld + col];
    } 

    inline const T_scal get_exp(idx row, idx col) const {
        if constexpr (L == Layout::ColMajor) {
            return shared_exps[(col*ld + row)/r];
        } else {
            return shared_exps[(row*ld + col)/r];
        }
    }

    // Added overload for linear indexing, used in transpose
    inline const T_scal get_exp(idx linear_index) const {
        return shared_exps[linear_index/r];
    }


    template<typename V>
    constexpr inline void set_exp(idx row, idx col, V value) const {

        if constexpr (L == Layout::ColMajor) {
            shared_exps[(col*ld + row)/r] = static_cast<shared_exp_type>(value);
        } else {
            shared_exps[(row*ld + col)/r] = static_cast<shared_exp_type>(value);
        }
    }

    // Added overload for linear indexing, used in transpose
    template<typename V>
    constexpr inline void set_exp(idx linear_index, V value) const {
        shared_exps[linear_index/r] = static_cast<shared_exp_type>(value);
    }

    template<typename V>
    constexpr inline V scaled_val(idx row, idx col) const {
        return operator()(row, col) * static_cast<V>(get_exp(row, col));
    }



    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((this)->template scaled_val<double>(row,col))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((this)->template scaled_val<double>(row,col))) return true;
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
concept is_Vanilla_MatrixType = requires {
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
template<typename T, typename T_scal, typename idx, Layout L>
struct is_MX_format<MX_Matrix<T, T_scal, idx, L>> {
    static constexpr bool value = true;
};
template<typename T, typename idx, Layout L>
struct is_MX_format<Matrix<T, idx, L>> {
    static constexpr bool value = false;
};

template<typename T>
concept AnyMatrixType = MatrixType<T> || MX_MatrixType<T>;

template<AnyMatrixType Mat_type>
using scalar_type = typename Mat_type::scalar_type;


template<MX_MatrixType Mat_type>
using shared_exp_type = typename Mat_type::shared_exp_type;




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
    using B_exp_type = typename DstMatrixType::shared_exp_type;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                // FIX: A is not an MX_Matrix and has no get_exp method.
                // Set exponent to 1.0 as a neutral value.
                B.set_exp(i,j, static_cast<B_exp_type>(1.0));
            }
        }
        
    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(1.0));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(1.0));
            }
        }
    }

    return;
    
}


template<AnyMatrixType SrcMatrixType, AnyMatrixType DstMatrixType>
inline void copy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) { return lacpy(A, B, uplo); }


template<MatrixType MatrixA, MatrixType MatrixA_t, int n_block_size = 1, int m_block_size = 1>
void transpose(MatrixA& A, MatrixA_t& At) {

    At.m = A.n;
    At.n = A.m;
    
    using At_type = typename MatrixA_t::scalar_type;
    
    //break into cases - if layouts are different, a simple data copy works. If same, block the transpose
    if constexpr (MatrixA::layout == MatrixA_t::layout) {
        // blocked transpose for same-layout
        int n_blocks = (At.rows() + n_block_size - 1) / n_block_size;
        int m_blocks = (At.cols() + m_block_size - 1) / m_block_size;

        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                // FIX: Corrected loop bounds to iterate over blocks in the destination matrix At.
                for(int row = b_row * n_block_size; row < std::min((b_row + 1) * n_block_size, (int)At.rows()); row++) {
                    for(int col = b_col * m_block_size; col < std::min((b_col + 1) * m_block_size, (int)At.cols()); col++) {
                            // At(row, col) = A(col, row)
                            At.data[At.get_idx(row, col)] = static_cast<At_type>(A.data[A.get_idx(col, row)]);
                    }
                }
            }
        }
    } else {
        // If layouts are different (e.g., ColMajor to RowMajor), a direct memory copy performs the transpose.
        for(int i = 0; i < A.rows()*A.cols(); i++) {
            At.data[i] = static_cast<At_type>(A.data[i]);
        }
    }
}


//for the case where At.r != A.r, just cast to a full precision matrix first and then cast back to a microsaled matrix
template<MX_MatrixType MX_MatrixA, MX_MatrixType MX_MatrixAt, int n_block_size = 1, int m_block_size = 1>
void transpose(MX_MatrixA& A, MX_MatrixAt& At)
{
    At.m = A.n;
    At.n = A.m;

    using At_type = typename MX_MatrixAt::scalar_type;

    //cases - if layouts are different just copy over. If same, block the transpose
    if constexpr (MX_MatrixA::layout == MX_MatrixAt::layout) {
        // blocked transpose for same-layout
        int n_blocks = (At.rows() + n_block_size - 1) / n_block_size;
        int m_blocks = (At.cols() + m_block_size - 1) / m_block_size;

        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                // FIX: Corrected loop bounds to iterate over blocks in the destination matrix At.
                for(int row = b_row * n_block_size; row < std::min((b_row + 1) * n_block_size, (int)At.rows()); row++) {
                    for(int col = b_col * m_block_size; col < std::min((b_col + 1) * m_block_size, (int)At.cols()); col++) {
                            // At(row, col) = A(col, row)
                            At.data[At.get_idx(row, col)] = static_cast<At_type>(A.data[A.get_idx(col, row)]);
                            // Copy exponent if it's the start of a block in the source matrix. Assumes A.r == At.r
                            if(A.get_idx(col, row) % A.r == 0) {
                                At.set_exp(row, col, A.get_exp(col, row));
                            }
                    }
                }
            }
        }
    } else {
        // FIX: If layouts are different, a direct memory copy of data and exponents performs the transpose.
        // The original implementation had incorrect function calls.
        // This assumes A.r == At.r
        size_t num_elements = (size_t)A.rows() * A.cols();
        for(size_t i = 0; i < num_elements; i++) {
            At.data[i] = static_cast<At_type>(A.data[i]);
            // Copy the corresponding shared exponent. The linear index overloads handle the r-grouping.
            At.set_exp(i, A.get_exp(i));
        }
    }
}

//slice to get submatrix
template<typename T, Layout L, typename idx, typename idx2, typename idx3>
Matrix<T, idx, L> slice(const Matrix<T, idx, L>& a, range<idx2> rows, range<idx3> cols) {
    assert(cols.first >= 0 && cols.second <= a.n && rows.first >= 0 && rows.second <= a.m);
    assert(cols.first <= cols.second && rows.first <= rows.second);

    // FIX: Starting pointer calculation must account for memory layout.
    idx offset;
    if constexpr (L == ColMajor) {
        offset = cols.first * a.ld + rows.first;
    }
    else { // RowMajor
        offset = rows.first * a.ld + cols.first;
    }
    
    return Matrix<T, idx, L>(a.data + offset,
                          rows.second - rows.first,
                          cols.second - cols.first,
                          a.ld);
}

template<typename T, typename T_scal, Layout L, typename idx, typename idx2, typename idx3>
MX_Matrix<T, T_scal, idx, L> slice(const MX_Matrix<T, T_scal, idx, L>& a, range<idx2> rows, range<idx3> cols) {
    assert(cols.first >= 0 && cols.second <= a.n && rows.first >= 0 && rows.second <= a.m);
    assert(cols.first <= cols.second && rows.first <= rows.second);

    // FIX: Starting pointer calculation must account for memory layout for both data and exponents.
    idx offset;
    if constexpr (L == ColMajor) {
        offset = cols.first * a.ld + rows.first;
    }
    else { // RowMajor
        offset = rows.first * a.ld + cols.first;
    }

    return MX_Matrix<T, T_scal, idx, L>(a.data + offset,
                                     a.shared_exps + offset / a.r,
                                     rows.second - rows.first,
                                     cols.second - cols.first,
                                     a.ld,
                                     a.r);
    }















} //namespace Lo_Gemm