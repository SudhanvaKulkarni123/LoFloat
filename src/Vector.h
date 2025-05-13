/// @author Sudhanva Kulkarni
/// Simple implementation of Vector and MXVector objects


#include <cassert>
#include "layouts.h"
#include <type_traits>


namespace Lo_Gemm {

template<typename idx>
using range = std::pair<idx, idx>;


template<typename T, typename idx>
class Vector {
    public :
    idx m;
    T* data;
    idx stride;

    Vector(T* data, idx m, idx stride = static_cast<idx>(1)) : data(data), m(m), stride(stride) {}

    T& constexpr inline  operator[] const (idx i) {
        return data[i*stride];
    }

    T& constexpr inline operator() const (idx i) {
        return data[i*stride];
    }   
    

};

template<typename T, typename idx, typename T_scal>
class MX_Vector {
    public:
    idx m;  //length of data vector
    idx n;  //length of shared_exps vector
    idx stride; //stride of data vector
    idx r; //number of contiguos elems that share common exp
    T_scal* shared_exps;
    T* data;

    MX_Vector(T* data, T_scal* shared_exps, idx m, idx n, idx stride = static_cast<idx>(1), idx r = static_cast<idx>(1)) : data(data), shared_exps(shared_exps), m(m), n(n), stride(stride), r(r) {}

    constexpr inline T& operator[] const (idx i) {
        return data[i*stride];
    }

    constexpr inline T& operator() const (idx i) {
        return data[i*stride] * shared_exps[i/r1];
    }

    constexpr inline T& get_exp(idx i) const {
        return shared_exps[i/r1];
    }

    constexpr inline T& set_exp(idx i, T_scal value) const {
        shared_exps[i/r1] = value;
    }

    constexpr inline idx length() const {
        return m;
    }

    constexpr inline idx length_exp() const {
        return n;
    }



};



template<typename T, typename idx>
T dot(const Vector<T, idx>& a, const Vector<T, idx>& b) {
    assert(a.m == b.m);
    T sum = 0;
    for (idx i = 0; i < a.m; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template<typename T, typename idx, typename T_scal, typename return_type = float>
return_type dot(const MX_Vector<T, idx, T_scal>& a, const MX_Vector<T, idx, T_scal>& b) {
    assert(a.m == b.m);
    return_type sum = return_type{};
    for (idx i = 0; i < a.r; ++i) {
        sum += a.get_exp(i) * b.get_exp(i) * a[i] * b[i];
    }
    return sum;
}

template<typename T1, typename idx1, typename T2, typename idx2>
void copy(const Vector<T1, idx1>& a, Vector<T2, idx2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            b[i] = static_cast<T2>(a[i]);
        }
        return;
    }
    


template<typename T1, typename idx1, typename T_scal1, typename T2, typename idx2, typename T_scal2>
void copy(const MX_Vector<T1, idx1, T_scal1>& a, MX_Vector<T2, idx2, T_scal2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            b[i] = static_cast<T2>(a[i]);
            if(i%b.r == 0) b.set_exp(i, static_cast<T_scal2>(a.get_exp(i)));
        }
        return;
    }



template<typename T1, typename idx, typename T2, typename idx2, typename T_scal>
void copy(const Vector<T1, idx>& a, MX_Vector<T2, idx2, T_scal>& b) {
    assert(a.m == b.m);
    int cnt = 0;
    T_scal exp = T_scal{};
    T1 maximum = T1{};
        for (int i = 0; i < a.m; ++i) {
           if(cnt == b.r) {
            if(std::is_integral_v<T_scal>) b.set_exp(i, static_cast<T_scal>(log2(maximum)));
            else b.set_exp(i, static_cast<T_scal>(maximum));
            cnt = 0;
            }
            maximum = std::max(maximum, a[i]);
            cnt++;
        }
        for(int i = 0; i < a.m; i++) {
            if(std::is_integral_v<T_scal>) b[i] = static_cast<T2>(a[i] / pow(2, b.get_exp(i)));
            else maximum = static_cast<T2>(a[i] / b.get_exp(i));

        }
        return;

}

template<typename T1, typename idx, typename T2, typename idx2, typename T_scal>
void copy(const MX_Vector<T1, idx, T_scal>& a, Vector<T2, idx2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            if(std::is_integeral_v<T_scal>) 
                b[i] = static_cast<T2>(a[i] * pow(2, exp));
            else b[i] = static_cast<T2>(a[i] * exp);
        }
        return;
}

template<typename T1, typename idx>
Vector<T1, idx> slice(const Vector<T1, idx>& a, range<idx> range) {
    assert(r.first >= 0 && r.second <= a.m);
    
    return Vector<T1, idx>(a.data + r.first * a.stride, r.second - r.first, a.stride);
}

template<typename T1, typename idx, typename T_scal>
MX_Vector<T1, idx, T_scal> slice(const MX_Vector<T1, idx, T_scal>& a, range<idx> range) {
    assert(r.first >= 0 && r.second <= a.m);
    
    return MX_Vector<T1, idx, T_scal>(a.data + r.first * a.stride, a.shared_exps + r.first * a.r, r.second - r.first, a.n - r.first * a.r, a.stride);
}



} //namespace Lo_gemm