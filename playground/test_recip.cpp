#include "lo_float.h"

int main() {

    using namespace lo_float;
    int counter = 0;
    int total = 256*256;

    for(int i = 0; i <  256; i++) {
        auto a = float8_ieee_p<4>::FromRep(i);
        for(int i = 0; i < 256; i++) {
            auto b = float8_ieee_p<4>::FromRep(i);
            auto f = a / b;
            

            //cmopute results exactly

            auto exact_f = static_cast<float>(a) / static_cast<float>(b);

            if(f == exact_f) {
                counter++;
            } 
        }
    }

    std::cout << "Passed " << counter << " out of " << total << " tests for float8_ieee_p<4> division." << std::endl;
    std::cout << "Pass rate: " << static_cast<double>(counter) / total << std::endl;
    return 0;
}