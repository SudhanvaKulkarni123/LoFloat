#include "lo_float.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <limits>


using namespace lo_float;

#include <cmath>
#define PI    3.14159265358979323846
#define PI_2  1.57079632679489661923   // PI / 2, rounded
#define PI_4  0.78539816339744830962   // PI / 4, rounded

template<typename U, typename T>
T poly_sine(T x_in)
{
    if (isinf(x_in) || isnan(x_in))
        return std::numeric_limits<T>::quiet_NaN();

    U x = static_cast<U>(x_in);               


    long long k = static_cast<long long>(std::floor( float((x + static_cast<U>(PI_4)) / static_cast<U>(PI_2) )));
    U r = x - static_cast<U>(k) * static_cast<U>(PI_2);

    //taylor
    U r2 = r * r;
    U r4 = r2 * r2;
    U r5 = r4 * r;
    U sin_r = r - (r * r2) / U(6.0) + (r5 / U(120.0));          
    U cos_r = U(1.0) - r2 / U(2.0) + (r4 / U(24.0));           

    switch (static_cast<int>(k & 3))
    {
        case 0:  return static_cast<T>( sin_r);   // 0⋅π/2 + r
        case 1:  return static_cast<T>( cos_r);   // 1⋅π/2 + r
        case 2:  return static_cast<T>(-sin_r);   // 2⋅π/2 + r
        default: return static_cast<T>(-cos_r);   // 3⋅π/2 + r
    }
}

int main() {
    //use ieee floats
    using float8_e4m3 = float8_ieee_p<4, Rounding_Mode::RoundToNearestEven, 4>;
    using fp16 = Templated_Float<halfPrecisionParams<Rounding_Mode::RoundToNearestEven>>;
    using bf16 = Templated_Float<bfloatPrecisionParams<Rounding_Mode::RoundToNearestEven>>;
    std::cout << std::scientific;
    std::cout << std::setprecision(10);
    //run the below loop for all values of p

    for(uint16_t i = 0; i < 128; i++) {
        float8_e4m3 a = float8_e4m3::FromRep((uint8_t)i);
        float8_e4m3 b = static_cast<float8_e4m3>(sin(static_cast<float>(a)));
        float8_e4m3 c = static_cast<float8_e4m3>(sin(static_cast<double>(a)));
        float8_e4m3 d = (poly_sine<float>(a));
        //print those cases where b != c
        if(d != c) {
            std::cout << "almost true value: " << sin(static_cast<double>(a)) << std::endl;
            std::cout << "value of a in double and float: " << (double)a << " " << (float)a << std::endl;
            std::cout <<  i << "." <<  " x: " << (float)a << " sin(x) using float: " << (float)b << ", sin(x) using double : " << (float)c << ", sin(x) wqith polynomial approx in fp16 : " << (float)d << std::endl;
        }
      
    }

    // using float8_e5m2 = float8_ieee_p<3, Rounding_Mode::RoundToNearestEven, 0>;
    // for(uint16_t i = 0; i < 128; i++) {
    //     float8_e5m2 a = float8_e5m2::FromRep((uint8_t)i);
    //     float8_e5m2 b = static_cast<float8_e5m2>(sin(static_cast<float>(a)));
    //     float8_e5m2 c = static_cast<float8_e5m2>(sin(static_cast<double>(a)));
    //     //print those cases where b != c
    //     if(b != c) {
    //         std::cout << "almost true value: " << sin(static_cast<double>(a)) << std::endl;
    //         std::cout << "value of a in double and float: " << (double)a << " " << (float)a << std::endl;
    //         std::cout <<  i << "." <<  " x: " << (float)a << " sin(x) using float: " << (float)b << ", sin(x) using double : " << (float)c << std::endl;
    //     }
      
    // }

    // using float8_e3m4 = float8_ieee_p<5, Rounding_Mode::RoundToNearestEven, 0>;
    // for(uint16_t i = 0; i < 128; i++) {
    //     float8_e3m4 a = float8_e3m4::FromRep((uint8_t)i);
    //     float8_e3m4 b = static_cast<float8_e3m4>(sin(static_cast<float>(a)));
    //     float8_e3m4 c = static_cast<float8_e3m4>(sin(static_cast<double>(a)));
    //     //print those cases where b != c
    //     if(b != c) {
    //         std::cout << "almost true value: " << sin(static_cast<double>(a)) << std::endl;
    //         std::cout << "value of a in double and float: " << (double)a << " " << (float)a << std::endl;
    //         std::cout <<  i << "." <<  " x: " << (float)a << " sin(x) using float: " << (float)b << ", sin(x) using double : " << (float)c << std::endl;
    //     }
      
    // }

    // using float8_e2m5 = float8_ieee_p<6, Rounding_Mode::RoundToNearestEven, 0>;
    // for(uint16_t i = 0; i < 128; i++) {
    //     float8_e2m5 a = float8_e2m5::FromRep((uint8_t)i);
    //     float8_e2m5 b = static_cast<float8_e2m5>(sin(static_cast<float>(a)));
    //     float8_e2m5 c = static_cast<float8_e2m5>(sin(static_cast<double>(a)));
    //     //print those cases where b != c
    //     if(b != c) {
    //         std::cout << "almost true value: " << sin(static_cast<double>(a)) << std::endl;
    //         std::cout << "value of a in double and float: " << (double)a << " " << (float)a << std::endl;
    //         std::cout <<  i << "." <<  " x: " << (float)a << " sin(x) using float: " << (float)b << ", sin(x) using double : " << (float)c << std::endl;
    //     }
      
    // }

    // using float8_e1m6 = float8_ieee_p<7, Rounding_Mode::RoundToNearestEven, 0>;
    // for(uint16_t i = 0; i < 128; i++) {
    //     float8_e1m6 a = float8_e1m6::FromRep((uint8_t)i);
    //     float8_e1m6 b = static_cast<float8_e1m6>(sin(static_cast<float>(a)));
    //     float8_e1m6 c = static_cast<float8_e1m6>(sin(static_cast<double>(a)));
    //     //print those cases where b != c
    //     if(b != c) {
    //         std::cout << "almost true value: " << sin(static_cast<double>(a)) << std::endl;
    //         std::cout << "value of a in double and float: " << (double)a << " " << (float)a << std::endl;
    //         std::cout <<  i << "." <<  " x: " << (float)a << " sin(x) using float: " << (float)b << ", sin(x) using double : " << (float)c << std::endl;
    //     }
      
    // }




    std::cout << "rounding\n";
    auto a = float8_e4m3(-1e-5);
    auto b = float8_e4m3(-1e-5f);

    std::cout << std::scientific;
  




    // std::cout << "justr a test" << std::endl;
    // std::cout << (float)float8_e4m3(1e-05) << std::endl;

    return 0;


}