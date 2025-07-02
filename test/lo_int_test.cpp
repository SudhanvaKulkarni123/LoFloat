///@author Sudhanva Kulkarni, UC Berkeley
#include <iostream>
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding random numbers
#include "lo_int.h"  

using namespace std;
using namespace lo_float;

int main() {
    srand(time(nullptr));  // Seed random number generator

    for (int i = 0; i < 10; i++) {  // Run 10 test cases
        // Generate random 4-bit unsigned and signed integers
        uint8_t rand_u = rand() % 16;  // [0, 15]
        int8_t rand_s = (rand() % 16) - 8;  // [-8, 7] for int4

        // using int4 = lo_float::int_n<4>;
        // using uint4 = lo_float::uint_n<4>;

        // Cast to uint4 and int4
        uint4 a = static_cast<uint4>(rand_u);
        uint4 b = static_cast<uint4>(rand() % 16);

        int4 c = static_cast<int4>(rand_s);
        int4 d = static_cast<int4>((rand() % 16) - 8);

        // Print values
        cout << "Test " << i + 1 << ":\n";
        cout << "  uint4: " << static_cast<int>(a) << " and " << static_cast<int>(b) << "\n";
        cout << "  int4 : " << static_cast<int>(c) << " and " << static_cast<int>(d) << "\n";

        // Perform arithmetic operations
        cout << "  uint4  -> Add: " << static_cast<int>(a + b)
             << ", Sub: " << static_cast<int>(a - b)
             << ", Mul: " << static_cast<int>(a * b)
             << ", Div: " << (b != (uint4)0 ? static_cast<int>(a / b) : -1) << "\n";

        cout << "  int4   -> Add: " << static_cast<int>(c + d)
             << ", Sub: " << static_cast<int>(c - d)
             << ", Mul: " << static_cast<int>(c * d)
             << ", Div: " << (d != (int4)0 ? static_cast<int>(c / d) : -1) << "\n";

        cout << "--------------------------------\n";
    }

    return 0;
}
