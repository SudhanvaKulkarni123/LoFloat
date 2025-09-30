/// @author Sudhanva Kulkarni
/// Simple implementation of Vector and MXVector objects


#include <cassert>
#include "lo_float.h"
#include "layouts.h"
#include <type_traits>


using namespace lo_float;

namespace Lo_Gemm {




template<Float T, Int idx = int>
class Vector {
    public :
    idx m;
    T* data;
    idx stride;
    using vector_type_tag = void;

    Vector(T* data, idx m, idx stride = static_cast<idx>(1)) : data(data), m(m), stride(stride) {}

    inline  T& operator[]  (idx i) const {
        return data[i*stride];
    }

    inline T& operator() (idx i) const {
        return data[i*stride];
    }   

    inline const idx len() {
        return m;
    }
    

};

template<Float T, Float T_scal, Int idx = int>
class MX_Vector {
    public:
    idx m;  //length of data vector
    idx n;  //length of shared_exps vector
    idx stride; //stride of data vector
    idx r; //number of contiguos elems that share common exp
    T_scal* shared_exps;
    T* data;
    using mx_vector_type_void = void;

    MX_Vector(T* data, T_scal* shared_exps, idx m, idx n, idx stride = static_cast<idx>(1), idx r = static_cast<idx>(1)) : data(data), shared_exps(shared_exps), m(m), n(n), stride(stride), r(r) {}

    inline T& operator[] (idx i) const {
        return data[i*stride];
    }

    inline T& operator() (idx i) const {
        return data[i*stride] * shared_exps[i/r];
    }

    inline const T& get_exp(idx i) const {
        return shared_exps[i/r];
    }

    inline const T& set_exp(idx i, T_scal value) const {
        shared_exps[i/r] = value;
    }

    constexpr inline idx len() const {
        return m;
    }

    constexpr inline idx len_exp() const {
        return n;
    }



};





template<Float T1, Int idx1, Float T2, Int idx2>
void copy(const Vector<T1, idx1>& a, Vector<T2, idx2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            b[i] = static_cast<T2>(a[i]);
        }
        return;
    }
    


template<Float T1, Int idx1, Float T_scal1, Int T2, Float idx2, Int T_scal2>
void copy(const MX_Vector<T1, idx1, T_scal1>& a, MX_Vector<T2, idx2, T_scal2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            b[i] = static_cast<T2>(a[i]);
            if(i%b.r == 0) b.set_exp(i, static_cast<T_scal2>(a.get_exp(i)));
        }
        return;
    }



template<Float T1, Int idx, Float T2, Int idx2, Float T_scal>
void copy(const Vector<T1, idx>& a, MX_Vector<T2, idx2, T_scal>& b) {
    assert(a.m == b.m);
    int cnt = 0;
    T_scal exp = T_scal{};
    T1 maximum = T1{};
        for (int i = 0; i < a.m; ++i) {
           if(cnt == b.r) {
                b.set_exp(i, static_cast<T_scal>(maximum));
                cnt = 0;
                for(int j = 0; j < b.r; j++) {
                    b[i+j] = static_cast<T2>((a[i+j] / b.get_exp(i)));
                }
            }
            maximum = maximum > a[i] ? maximum : a[i];
            cnt++;
        }
        return;

}

template<Float T1, Int idx, Float T2, Int idx2, Float T_scal>
void copy(const MX_Vector<T1, idx, T_scal>& a, Vector<T2, idx2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            b[i] = static_cast<T2>(a[i] * a.get_exp(i));
        }
        return;
}

template<Float T1, Int idx>
Vector<T1, idx> slice(const Vector<T1, idx>& a, range<idx> range) {
    assert(range.first >= 0 && range.second <= a.m && range.first <= range.second);
    
    return Vector<T1, idx>(a.data + range.first * a.stride, range.second - range.first, a.stride);
}

template<Float T1, Int idx, Float T_scal>
MX_Vector<T1, idx, T_scal> slice(const MX_Vector<T1, idx, T_scal>& a, range<idx> range) {
    assert(range.first >= 0 && range.second <= a.m);
    
    return MX_Vector<T1, idx, T_scal>(a.data + range.first * a.stride, a.shared_exps + range.first * a.r, range.second - range.first, a.n - range.first * a.r, a.stride);
}

//converts normal vector<t1, idx1> to vector<T2, idx2> while simulating the effects of microscaling
template<Float T1, Int idx1, Float T2, Int idx2, Float T_scal, Float T_buf>
void fake_mxcopy(const Vector<T1, idx1>& a, Vector<T2, idx2>& b)
{
    assert(a.m == b.m);
    int cnt = 0;
    T_scal exp = T_scal{};
    T1 maximum = T1{};
        for (int i = 0; i < a.m; ++i) {
           if(cnt == b.r) {
            exp = static_cast<T_scal>(maximum);
            for(int j = 0; j < b.r; j++) {
                b[i+j] = static_cast<T2>(static_cast<T_buf>(a[i+j] / exp)) * static_cast<T2>(exp);
            }
            cnt = 0;
            maximum = T1{};
        }
        maximum = maximum > abs(a[i]) ? maximum : abs(a[i]);
        cnt++;
    }
        return;

}


// Trait: true if T has MX_matrix_type_tag
template<typename T>
struct is_VectorType {
    static constexpr bool value = requires { typename T::vector_type_tag; };
};

template<typename T>
using is_VectorType_t = typename is_VectorType<T>::value;

// // Concept: matches only if T has matrix_type_tag (Vanilla Matrix)
// template<typename T>
// concept is_Vanilla_MatrixType = requires {
//     typename T::matrix_type_tag;
// };

// template<typename T>    
// struct is_Vanilla_MatrixType_t {
//     static constexpr bool value = requires { typename T::matrix_type_tag; };
// };

// //helper that returns if the format is MX by checking if the type has a shared_exps member
// template<typename T>
// struct is_MX_format {
//     static constexpr bool value = false;
// };

// template<typename T, typename T_scal, typename idx, Layout L>
// struct is_MX_format<MX_Matrix<T, T_scal, idx, L>> {
//     static constexpr bool value = true;
// };

// template<typename T, typename idx, Layout L>
// struct is_MX_format<Matrix<T, idx, L>> {
//     static constexpr bool value = false;
// };

// template<typename T>
// concept AnyMatrixType = MatrixType<T> || MX_MatrixType<T>;



} //namespace Lo_gemm