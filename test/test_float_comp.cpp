#include "lo_float.h"
#include <iostream>
#include <iomanip>


int main() {
    using fp8 = float8_ieee_p<4, Rounding_Mode::RoundToNearestEven, 4>;
    for(unsigned int i = 0; i < 128; i++) {

        for(unsigned int j = i; j < 128; j++) {
            fp8 a = fp8::FromRep(i);
            fp8 b = fp8::FromRep(j);
            
            if(!(b >= a)) {
                std::cout << "i: " << i << ", j: " << j << ", a: " << (double)a << ", b: " << (double)b << std::endl;
            }
        }
        
    }
}