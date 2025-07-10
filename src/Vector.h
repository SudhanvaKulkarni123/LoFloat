/// @author Sudhanva Kulkarni
/// Simple implementation of Vector and MXVector objects


#include <cassert>
#include "lo_float.h"
#include <type_traits>


using namespace lo_float;

namespace Lo_Gemm {

template<Int idx>
using range = std::pair<idx, idx>;


template<Float T, Int idx>
class Vector {
    public :
    idx m;
    T* data;
    idx stride;

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

template<Float T, Int idx, Float T_scal>
class MX_Vector {
    public:
    idx m;  //length of data vector
    idx n;  //length of shared_exps vector
    idx stride; //stride of data vector
    idx r; //number of contiguos elems that share common exp
    T_scal* shared_exps;
    T* data;

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
            if(std::is_integral_v<T_scal>) b.set_exp(i, static_cast<T_scal>((maximum)));
            else b.set_exp(i, static_cast<T_scal>(maximum));
            cnt = 0;
            }
            maximum = maximum > a[i] ? maximum : a[i];
            cnt++;
        }
        for(int i = 0; i < a.m; i++) {
            if(std::is_integral_v<T_scal>) b[i] = static_cast<T2>(a[i] / b.get_exp(i));
            else maximum = static_cast<T2>(a[i] / b.get_exp(i));

        }
        return;

}

template<Float T1, Int idx, Float T2, Int idx2, Float T_scal>
void copy(const MX_Vector<T1, idx, T_scal>& a, Vector<T2, idx2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            if(std::is_integral_v<T_scal>) 
                b[i] = static_cast<T2>(a[i] * a.get_exp(i));
            else b[i] = static_cast<T2>(a[i] * a.get_exp(i));
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



} //namespace Lo_gemm