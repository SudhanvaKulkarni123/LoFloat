/// @author Sudhanva Kulkarni
/// Simple implementation of Vector and MXVector objects


#include <cassert>
#include "layouts.h"


namespace Lo_Gemm {

template<typename T, typename idx>
class Vector {
    private :
    idx m;
    T* data;

}

template<typename T, typename idx, typename T_scal>
class MX_Vector {
    private:
    idx m;
    idx n;
    idx ld;
    idx r1; 
    


}

} //namespace Lo_gemm