import LoFloat as lof
class P3109_NaNChecker:
    def __init__(self, k, is_signed=True):
        self.k = k
        self.is_signed = is_signed
    
    def __call__(self, bits):
        if self.is_signed:
            return bits == (1 << (self.k - 1))
        else:
            return bits == ((1 << self.k) - 1)
    
    def qNanBitPattern(self):
        if self.is_signed:
            return 1 << (self.k - 1)
        else:
            return (1 << self.k) - 1
    
    def sNanBitPattern(self):
        if self.is_signed:
            return 1 << (self.k - 1)
        else:
            return (1 << self.k) - 1


class P3109_InfChecker:
    def __init__(self, k, is_signed=True, has_inf_saturating=True):
        self.k = k
        self.is_signed = is_signed
        self.has_inf_saturating = has_inf_saturating
    
    def __call__(self, bits):
        if not self.has_inf_saturating:
            if self.is_signed:
                target = (1 << self.k) - 1
            else:
                target = (1 << self.k) - 2
            return (bits | (1 << (self.k - 1))) == target
        else:
            return False
    
    def minNegInf(self):
        if self.is_signed:
            return (1 << self.k) - 1
        else:
            return 0
    
    def minPosInf(self):
        if self.is_signed:
            return (1 << (self.k - 1)) - 1
        else:
            return (1 << self.k) - 2
        

def create_p3109_params(k, p, is_signed=True, saturating=True, bias=None):
    mantissa_bits = p - 1
    if bias is None:
        bias = 1 << (k - p - 1)
    inf_behavior = lof.InfBehavior.Saturating if saturating else lof.InfBehavior.Extended
   
    
    return lof.FloatFormatDescriptor(
        k,
        mantissa_bits,
        bias,
        inf_behavior=inf_behavior,
        nan_behavior=lof.NaNBehavior._3109,
        signedness=lof.Signedness.Signed if is_signed else lof.Signedness.Unsigned,
        is_inf_checker=P3109_InfChecker(k, is_signed, saturating),
        is_nan_checker=P3109_NaNChecker(k, is_signed)
    )


def create_e8m0_params():
    """E8M0 shared-scale format for microscaling (MX): 8-bit unsigned, 0 mantissa
    bits (power-of-two), bias 127 -> range 2^-127 .. 2^128. This is the standard
    MX scale format; pass it as the `scale_format` of virtual_mx_round. (The MX
    path ignores the inf/nan checkers, so the P3109 ones below are placeholders.)"""
    return lof.FloatFormatDescriptor(
        8,
        0,
        127,
        inf_behavior=lof.InfBehavior.Extended,
        nan_behavior=lof.NaNBehavior._3109,
        signedness=lof.Signedness.Unsigned,
        is_inf_checker=P3109_InfChecker(8, False, False),
        is_nan_checker=P3109_NaNChecker(8, False),
    )



class HalfPrecisionInfChecker:
    def __call__(self, bits):
        exp_mask = 0x7C00
        mant_mask = 0x03FF
        return (bits & exp_mask) == exp_mask and (bits & mant_mask) == 0
    
    def minNegInf(self):
        return 0xFC00
    
    def minPosInf(self):
        return 0x7C00


class HalfPrecisionNaNChecker:
    def __call__(self, bits):
        exp_mask = 0x7C00
        mant_mask = 0x03FF
        return (bits & exp_mask) == exp_mask and (bits & mant_mask) != 0
    
    def qNanBitPattern(self):
        return 0x7E00
    
    def sNanBitPattern(self):
        return 0x7D00

def create_half_params():
    return lof.FloatFormatDescriptor(
        16,
        10,
        15,
        inf_behavior=lof.InfBehavior.Extended,
        nan_behavior=lof.NaNBehavior._3109,
        signedness=lof.Signedness.Signed,
        is_inf_checker=HalfPrecisionInfChecker(),
        is_nan_checker=HalfPrecisionNaNChecker()
    )

class SinglePrecisionInfChecker:
    def __call__(self, bits):
        exp_mask  = 0x7F800000
        mant_mask = 0x007FFFFF
        return (bits & exp_mask) == exp_mask and (bits & mant_mask) == 0

    def minNegInf(self):
        return 0xFF800000

    def minPosInf(self):
        return 0x7F800000


class SinglePrecisionNaNChecker:
    def __call__(self, bits):
        exp_mask  = 0x7F800000
        mant_mask = 0x007FFFFF
        return (bits & exp_mask) == exp_mask and (bits & mant_mask) != 0

    def qNanBitPattern(self):
        return 0x7FC00000   # quiet NaN (most significant mantissa bit set)

    def sNanBitPattern(self):
        return 0x7FA00000   # signaling NaN


def create_single_params():
    return lof.FloatFormatDescriptor(
        32,          
        23,          
        127,          
        inf_behavior=lof.InfBehavior.Extended,
        nan_behavior=lof.NaNBehavior._3109,
        signedness=lof.Signedness.Signed,
        is_inf_checker=SinglePrecisionInfChecker(),
        is_nan_checker=SinglePrecisionNaNChecker()
    )
