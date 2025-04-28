#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
size_t get_cache_size(int level) {
    char path[128];
    snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/size", level);
    FILE* fp = fopen(path, "r");
    if (!fp) return 0;

    char buffer[32];
    if (!fgets(buffer, sizeof(buffer), fp)) {
        fclose(fp);
        return 0;
    }
    fclose(fp);

    size_t size = 0;
    if (strchr(buffer, 'K')) {
        sscanf(buffer, "%zuK", &size);
        size *= 1024;
    } else if (strchr(buffer, 'M')) {
        sscanf(buffer, "%zuM", &size);
        size *= 1024 * 1024;
    }
    return size;
}

size_t get_cache_line_size() {
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
    if (!fp) return 0;

    size_t size = 0;
    fscanf(fp, "%zu", &size);
    fclose(fp);
    return size;
}

#elif defined(__APPLE__)
#include <sys/sysctl.h>

size_t get_cache_size(int level) {
    int mib[2];
    size_t size = 0;
    size_t len = sizeof(size);
    switch (level) {
        case 0: mib[0] = CTL_HW; mib[1] = HW_L1ICACHESIZE; break;
        case 1: mib[0] = CTL_HW; mib[1] = HW_L2CACHESIZE; break;
        case 2: mib[0] = CTL_HW; mib[1] = HW_L3CACHESIZE; break;
        default: return 0;
    }
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0)
        return size;
    return 0;
}

size_t get_cache_line_size() {
    size_t line_size = 0;
    size_t size = sizeof(line_size);
    if (sysctlbyname("hw.cachelinesize", &line_size, &size, 0, 0) != 0)
        return 0;
    return line_size;
}
#else
size_t get_cache_size(int level) { return 0; }
size_t get_cache_line_size() { return 0; }
#endif

// Get the system page size (TLB line size)
size_t get_page_size() {
    return sysconf(_SC_PAGESIZE);  // Usually 4096 bytes
}

// Stub for TLB size (platform-dependent)
size_t get_TLB_size(int level) {
#if defined(__linux__) && defined(__x86_64__)
    unsigned int eax, ebx, ecx, edx;
    __cpuid(2, eax, ebx, ecx, edx);
    unsigned int descriptors[4] = {eax, ebx, ecx, edx};
    printf("TLB descriptors (raw):\n");
    for (int i = 0; i < 4; i++) {
        if (descriptors[i] & (1 << 31)) continue; // invalid
        for (int j = 0; j < 4; j++) {
            unsigned char desc = (descriptors[i] >> (j * 8)) & 0xFF;
            if (desc == 0) continue;
            printf("  0x%X\n", desc);
        }
    }
    return 0; // actual size must be interpreted from descriptor tables
#else
    return 0;
#endif
}



//test function for rate of streaming from L2 cache
double L2_streaming_rate() {
constexpr size_t N = 8 * 1024 * 1024; // 8M floats = 32MB (fits in L2 cache for many CPUs)
std::vector<float> data(N, 1.0f);

// Warm-up: read all the data to ensure it's in L2
volatile float sink = 0.0f;
for (size_t i = 0; i < N; ++i) {
    sink += data[i];
}

// Real measurement
uint64_t start_cycles = rdtsc();

float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for (size_t i = 0; i < N; i += 4) {
    sum0 += data[i + 0];
    sum1 += data[i + 1];
    sum2 += data[i + 2];
    sum3 += data[i + 3];
}

uint64_t end_cycles = rdtsc();

float total = sum0 + sum1 + sum2 + sum3;

uint64_t elapsed_cycles = end_cycles - start_cycles;
double cycles_per_element = static_cast<double>(elapsed_cycles) / N;

return cycles_per_element;
}