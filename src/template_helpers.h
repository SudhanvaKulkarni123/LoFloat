

namespace lo_float {
    namespace lo_float_internal {

//--------------------------------- mantissa bits-----------------------------

template <class T, class = void>
struct get_mantissa_bits {                       // default: binary64
    static constexpr int value = 53;
};

template <class T>
struct get_mantissa_bits<T, std::void_t<decltype(T::mantissa_bits)>> {
    static constexpr int value = T::mantissa_bits;
};

template <class T>
inline constexpr int get_mantissa_bits_v = get_mantissa_bits<T>::value;


//-------------------------------- stochastic‑rounding length ------------------------------------

template <class T, class = void>
struct get_stochastic_length {                   // default: none
    static constexpr int value = 0;
};

template <class T>
struct get_stochastic_length<T,
          std::void_t<decltype(T::stochastic_rounding_length)>> {
    static constexpr int value = T::stochastic_rounding_length;
};

template <class T>
inline constexpr int get_stochastic_length_v = get_stochastic_length<T>::value;


//-------------------------------- signed / unsigned info ----------------------------------------

template <class T, class = void>
struct get_signedness {                          // default: signed
    static constexpr lo_float::Signedness value =
        lo_float::Signedness::Signed;
};

template <class T>
struct get_signedness<T, std::void_t<decltype(T::is_signed)>> {
    static constexpr lo_float::Signedness value = T::is_signed;
};

template <class T>
inline constexpr lo_float::Signedness get_signedness_v =
    get_signedness<T>::value;


//-------------------------------- unsigned‑value behaviour --------------------------------------

template <class T, class = void>
struct get_unsigned_behavior {                   // default: negative → 0
    static constexpr lo_float::Unsigned_behavior value =
        lo_float::Unsigned_behavior::NegtoZero;
};

template <class T>
struct get_unsigned_behavior<T,
          std::void_t<decltype(T::unsigned_behavior)>> {
    static constexpr lo_float::Unsigned_behavior value =
        T::unsigned_behavior;
};

template <class T>
inline constexpr lo_float::Unsigned_behavior get_unsigned_behavior_v =
    get_unsigned_behavior<T>::value;


//-------------------------------- NaN behaviour --------------------------------------------------

template <class T, class = void>
struct get_NaN_Behavior {                        // default: quiet NaN
    static constexpr lo_float::NaN_Behaviors value =
        lo_float::NaN_Behaviors::QuietNaN;
};

template <class T>
struct get_NaN_Behavior<T, std::void_t<decltype(T::NaN_behavior)>> {
    static constexpr lo_float::NaN_Behaviors value = T::NaN_behavior;
};

template <class T>
inline constexpr lo_float::NaN_Behaviors get_NaN_Behavior_v =
    get_NaN_Behavior<T>::value;


//-------------------------------- overflow / ±Inf behaviour -------------------------------------

template <class T, class = void>
struct get_overflow_behavior {                   // sensible default
    static constexpr lo_float::Inf_Behaviors value =
        lo_float::Inf_Behaviors::Extended;
};

template <class T>
struct get_overflow_behavior<T,
          std::void_t<decltype(T::Overflow_behavior)>> {
    static constexpr lo_float::Inf_Behaviors value =
        T::Overflow_behavior;
};

template <class T>
inline constexpr lo_float::Inf_Behaviors get_overflow_behavior_v =
    get_overflow_behavior<T>::value;


//-------------------------------- rounding mode --------------------------------------------------

template <class T, class = void>
struct get_rounding_mode {                       // default: ties‑to‑even
    static constexpr lo_float::Rounding_Mode value =
        lo_float::Rounding_Mode::RoundToNearestEven;
};

template <class T>
struct get_rounding_mode<T, std::void_t<decltype(T::rounding_mode)>> {
    static constexpr lo_float::Rounding_Mode value = T::rounding_mode;
};

template <class T>
inline constexpr lo_float::Rounding_Mode get_rounding_mode_v =
    get_rounding_mode<T>::value;


//-------------------------------- bit‑width ------------------------------------------------------

template <class T, class = void>
struct get_bitwidth {                            // default: 32 bits
    static constexpr int value = 32;
};

template <class T>
struct get_bitwidth<T, std::void_t<decltype(T::bitwidth)>> {
    static constexpr int value = T::bitwidth;
};

template <class T>
inline constexpr int get_bitwidth_v = get_bitwidth<T>::value;


//-------------------------------- exponent bias -------------------------------------------------

template <class T, class = void>
struct get_bias {                                // default: 0
    static constexpr int value = 0;
};

template <class T>
struct get_bias<T, std::void_t<decltype(T::bias)>> {
    static constexpr int value = T::bias;
};

template <class T>
inline constexpr int get_bias_v = get_bias<T>::value;




// pointers to NaN and Inf checkers 

template <class T, class = void>
struct get_IsNaN {                              // default: no NaN
    static constexpr auto value = nullptr;
};

template <class T>
struct get_IsNaN<T, std::void_t<decltype(T::IsNaNFunctor)>> {
    static constexpr auto value = &T::IsNaNFunctor;
};

template <class T>
inline constexpr auto get_IsNaN_v = get_IsNaN<T>::value;

template <class T, class = void>
struct get_IsInf {                              // default: no Inf
    static constexpr auto value = nullptr;
};

template <class T>
struct get_IsInf<T, std::void_t<decltype(T::IsInfFunctor)>> {
    static constexpr auto value = &T::IsInfFunctor;
};

template <class T>
inline constexpr auto get_IsInf_v = get_IsInf<T>::value;





}   //namespace lo_float_internal


template<typename T>
struct get_len {
    static constexpr int val = 0;
};

template<>
struct get_len<int> {
    static constexpr int val = 32;
};

template<int len, Signedness sign>
struct get_len<i_n<len, sign>> {
    static constexpr int val = len;
};

template<typename T>
inline constexpr auto get_int_len_v = get_len<T>::val;


template<typename T>
inline constexpr auto get_mantissa_bits_v = lo_float_internal::get_mantissa_bits_v<T>;

template<typename T>
inline constexpr auto get_bias = lo_float_internal::get_bias_v<T>;
}   //namespace lo_float