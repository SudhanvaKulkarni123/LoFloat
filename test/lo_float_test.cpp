#include <stdio.h>
#include <iostream>
#include <bitset>
#include <stdlib.h>
#include <cstdint>
#include "lo_float.h"
#include "lo_float_sci.hpp"


using namespace lo_float;

int main() {

    using fp4 = float4_p<2>;
    using fp6 = float6_p<3>;
    using fp8 = P3109_float<8, 4, Signedness::Signed, Inf_Behaviors::Extended>;


    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.5, 0.1);

    float r1 = dist(gen);
    float r2 = dist(gen);
    float r3 = dist(gen);
    float r4 = dist(gen);

    std::cout << "Random numbers r1, r2, r3, r4: \n";
    std::cout << r1 << ", " << r2 << ", " << r3 << ", " << r4 << "\n";

    std::cout << "\nTesting arithmetic operations in different floating-point precisions:\n";

    std::cout << "Addition:\n";
    // Addition
    for(int i = 0;i < 10; i++) {
        std::cout << " fp4:  " << (float)static_cast<fp4>(r1) << " + " << (float)static_cast<fp4>(r2) << " = " 
        << (float)static_cast<fp4>(r1) + (float)static_cast<fp4>(r2) << "\n";
        std::cout << " fp6:  " << (float)static_cast<fp6>(r1) << " + " << (float)static_cast<fp6>(r2) << " = " 
                << (float)static_cast<fp6>(r1) + (float)static_cast<fp6>(r2) << "\n";
        std::cout << " fp8:  " << (float)static_cast<fp8>(r1) << " + " << (float)static_cast<fp8>(r2) << " = " 
                << (float)static_cast<fp8>(r1) + (float)static_cast<fp8>(r2) << "\n";
        std::cout << " fp32: " << r1 << " + " << r2 << " = " << r1 + r2 << "\n";
    }


    // Multiplication
    std::cout << "\nMultiplication:\n";
    for(int i = 0;i < 10; i++) {
    std::cout << " fp4:  " << (float)static_cast<fp4>(r3) << " * " << (float)static_cast<fp4>(r4) << " = " 
              << (float)static_cast<fp4>(r3) * (float)static_cast<fp4>(r4) << "\n";
    std::cout << " fp6:  " << (float)static_cast<fp6>(r3) << " * " << (float)static_cast<fp6>(r4) << " = " 
              << (float)static_cast<fp6>(r3) * (float)static_cast<fp6>(r4) << "\n";
    std::cout << " fp8:  " << (float)static_cast<fp8>(r3) << " * " << (float)static_cast<fp8>(r4) << " = " 
              << (float)static_cast<fp8>(r3) * (float)(float)static_cast<fp8>(r4) << "\n";
    std::cout << " fp32: " << r3 << " * " << r4 << " = " << r3 * r4 << "\n";
    }
    // Division
    std::cout << "\nDivision:\n";
    for(int i = 0;i < 10; i++) {
    std::cout << " fp4:  " << (float)(float)static_cast<fp4>(r1) << " / " << (float)(float)static_cast<fp4>(r3) << " = " 
              << (float)((float)static_cast<fp4>(r1) /(float)static_cast<fp4>(r3)) << "\n";
    std::cout << " fp6:  " << (float)(float)static_cast<fp6>(r1) << " / " << (float)static_cast<fp6>(r3) << " = " 
              << (float)static_cast<fp6>(r1) / (float)static_cast<fp6>(r3) << "\n";
    std::cout << " fp8:  " << (float)static_cast<fp8>(r1) << " / " << (float)static_cast<fp8>(r3) << " = " 
              << (float)static_cast<fp8>(r1) / (float)static_cast<fp8>(r3) << "\n";
    std::cout << " fp32: " << r1 << " / " << r3 << " = " << r1 / r3 << "\n";
    }

    //Divide number by itself
    std::cout << "\nDivide number by itself:\n";
    for(int i = 0;i < 10; i++) {
    std::cout << " fp4:  " << (float)static_cast<fp4>(r1) << " / " << (float)static_cast<fp4>(r1) << " = " 
              << (float)static_cast<fp4>(r1) / (float)static_cast<fp4>(r1) << "\n";
    std::cout << " fp6:  " << (float)static_cast<fp6>(r1) << " / " << (float)static_cast<fp6>(r1) << " = " 
              << (float)static_cast<fp6>(r1) / (float)static_cast<fp6>(r1) << "\n";
    std::cout << " fp8:  " << (float)static_cast<fp8>(r1) << " / " << (float)static_cast<fp8>(r3) << " = " 
              << (float)static_cast<fp8>(r1) / (float)static_cast<fp8>(r1) << "\n";
    std::cout << " fp32: " << r1 << " / " << r1 << " = " << r1 / r1 << "\n";
    }





    std::cout << "generate fp8 numbers : \n";

    for(u_int8_t i = 0; i < 128;i++) {
        std::cout << (float)fp8::FromRep(i) << ",";
    }


    return 0;
}






