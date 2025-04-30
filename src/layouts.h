#include <cstdint>

namespace Lo_Gemm { 
enum Layout : uint8_t {
    ColMajor = 0,
    RowMajor = 1
};

enum MX_Layout : uint8_t {
    byColumn = 0,
    byRow = 1,
    byBlock = 2
};

enum Uplo : uint8_t {
    Upper = 0,
    Lower = 1, 
    General = 2
};

struct MX_tuple {
    MX_Layout layout;
    int m;
    int n;

    MX_tuple(MX_Layout layout, int m, int n = 1) : layout(layout), m(m), n(n) {}
};

enum Arch_extensions : uint8_t {
    NONE = 0,
    AVX256 = 1,
    AVX512 = 2,
    NEON = 3
};
}
