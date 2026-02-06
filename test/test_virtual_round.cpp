#include "lo_float.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>

int main() {
    int N = 4000;
    float arr[N];
    float og_arr[N];
    srand(42);

    for(int i = 0; i < N; ++i) {
        arr[i] = (static_cast<float>(rand()) / RAND_MAX)*1.0 ; 
        og_arr[i] = arr[i];
    }

    lo_float::virtual_round(arr, arr, N, lo_float::halfPrecisionParams, lo_float::Rounding_Mode::RoundToNearestEven, 0);

    if (N < 5) {
        for (int i = 0; i < N; ++i) {
            printf("Original: %f, Rounded: %f\n", og_arr[i], arr[i]);
            printf("bit patterns in hex: Original: %08x, Rounded: %08x\n", 
                   *((uint32_t*)&og_arr[i]), *((uint32_t*)&arr[i]));
        }
    } 
    //check err
    for (int i = 0; i < N; ++i) {
        float err = std::abs(arr[i] - og_arr[i]);
        int exp = 0;
        std::frexp(og_arr[i], &exp);         // d == mantissa * 2^exp
        float two_pow = std::ldexp(1.0, exp); // returns 1.0 * 2^exp
        if(err > std::pow(2.0, -11)*two_pow) {
            printf("Value: %f, Original: %f, Error: %f\n", arr[i], og_arr[i], err);
            printf("Test failed!\n");
            return -1;
        }
    }
}