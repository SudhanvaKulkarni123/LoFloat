///@author Sudhanva Kulkarni, Berkeley 
#include "lo_float_sci.hpp"
#include <iostream>
#include <cmath>

using namespace lo_float;

template<int k, int p>
bool F2I_test() {
    using fp = P3109_float<k, p, Signedness::Signed, Inf_Behaviors::Saturating>;
    using i_k = i_n<k + 2, Signedness::Signed>;

    for (uint16_t i = 0; i < (1 << (k-1)); i++) {
        fp a = fp::FromRep(i);
        i_k b = i_k(a);
        // Check that abs(b - a) <= 0.5
        if (fabs(float(b) - float(a)) > 0.5f) {
            std::cout << "F2I_test failed at i = " << i 
                      << ", a = " << float(a) 
                      << ", b = " << float(b) << "\n";

        }
    }
    return false;
}

template<int k, int p>
bool I2F_test() {
    using fp = P3109_float<k, p, Signedness::Signed, Inf_Behaviors::Saturating>;
    using i_k = i_n<k + 2, Signedness::Signed>;

    for (int16_t i = -(1 << k); i < (1 << k); i++) {
        i_k b = i_k(i);
        fp a = fp(b);
        if( abs(b) > (i_k) abs(std::numeric_limits<fp>::max())) {
            if(abs(a) != abs(std::numeric_limits<fp>::max())) {
                std::cout << "test failed due to incorrect saturation\n";

            }
        }
        if (fabs(float(b) - float(a)) > 0.5f) {
            std::cout << "I2F_test failed at i = " << i 
                      << ", a = " << float(a) 
                      << ", b = " << float(b) << "\n";

        }
    }
    return false;
}

int main() {
    std::cout << "Running int to float and float to int conversion tests\n";

    // std::cout << "checking conversion of -8 to double : " << (double) -8 << "\n";
    // std::cout << "checkinf conversion of -8 to e1m1 : " << (int)(P3109_float<3, 1, Signedness::Signed, Inf_Behaviors::Saturating>(-7)).rep() << "\n";
    // // std::cout << "printing all possible e1m1 values : " << std::endl;
    // // for(uint8_t i = 0; i < 8; i++) {
    // //     std::cout << (float) P3109_float<3, 1, Signedness::Signed, Inf_Behaviors::Saturating>::FromRep(i) << "\n";
    // // }
    // // std::cout << "chyeckinhg operator-\n";
    // // std::cout << (float)-P3109_float<3, 1, Signedness::Signed, Inf_Behaviors::Saturating>(7) << "\n";

    for (int k = 3; k <= 6; ++k) {
        
            std::cout << "Testing k = " << k << "...\n";

            bool f2i_fail = false;
            bool i2f_fail = false;

            switch (k) {
                case 3: i2f_fail = I2F_test<3, 1>(); break;
                case 4: i2f_fail = I2F_test<4, 1>(); break;
                case 5: i2f_fail = I2F_test<5, 1>(); break;
                case 6: i2f_fail = I2F_test<6, 1>(); break;
                default: break;
            }

            switch(k) {
                case 3: f2i_fail = F2I_test<3, 1>(); break;
                case 4: f2i_fail = F2I_test<4, 1>(); break;
                case 5: f2i_fail = F2I_test<5, 1>(); break;
                case 6: f2i_fail = F2I_test<6, 1>(); break;
                default: break;
            }

            if (f2i_fail || i2f_fail)
                std::cout << "❌ Test FAILED for k = " << k << "\n";
            else
                std::cout << "✅ Test PASSED for k = " << k <<   "\n";
        
    }

    return 0;
}
