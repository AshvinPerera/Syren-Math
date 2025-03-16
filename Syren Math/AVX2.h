#pragma once

#include <immintrin.h>

/**
 * @brief A collection of AVX2-based SIMD utility functions for optimized numerical operations.
 *
 * @details
 * This header file provides a set of highly efficient, low-level functions for performing SIMD operations using AVX2 intrinsics.
 * It includes functions for aligned memory operations, arithmetic operations, fused multiply-add operations, reduction operations,
 * and mask generation for variable-sized SIMD operations.
 */

const unsigned short CMAXDOUBLE = 4;
const unsigned short CMAXINT64 = 4;
const unsigned short CMAXFLOAT = 8;
const unsigned short CMAXINT32 = 8;
const unsigned short CMAXINT16 = 16;
const unsigned short CMAXINT8 = 32;

/** Stores a 256-bit integer register into aligned memory.
 * 
 * @tparam T1 Pointer type (must be `__m256i*`).
 * @tparam T2 Value type (`__m256i`).
 * @param pointer Pointer to store the value.
 * @param value Value to store.
 */
template<typename T1, typename T2>
constexpr auto StoreInt(T1 pointer, T2  value) { return _mm256_store_si256(pointer, value); }

/** Stores a 256-bit integer register into aligned memory using a mask representing 32-bit integers.
 *
 * @tparam T1 Pointer type (must be `__m256i*`).
 * @tparam T2 Mask type (`__m256i`).
 * @tparam T3 Value type (`__m256i`).
 * @param pointer Pointer to store the value.
 * @param mask Mask to apply to the store operation.
 * @param value Value to store.
 */
template<typename T1, typename T2, typename T3>
constexpr auto StoreMaskInt32(T1 pointer, T2  mask, T3 value) { return _mm256_maskstore_epi32(pointer, mask, value); }

/** Stores a 256-bit integer register into aligned memory using a mask representing 64-bit integers.
 *
 * @tparam T1 Pointer type (must be `__m256i*`).
 * @tparam T2 Mask type (`__m256i`).
 * @tparam T3 Value type (`__m256i`).
 * @param pointer Pointer to store the value.
 * @param mask Mask to apply to the store operation.
 * @param value Value to store.
 */
template<typename T1, typename T2, typename T3>
constexpr auto StoreMaskInt64(T1 pointer, T2  mask, T3 value) { return _mm256_maskstore_epi64(pointer, mask, value); }

/** Loads a 256-bit integer register from aligned memory.
 *
 * @tparam T Pointer type (must be `__m256i*`).
 * @param pointer Pointer to load the value from.
 */
template<typename T>
constexpr auto LoadInt(T pointer) { return _mm256_load_si256(pointer); }

/** Loads a 256-bit integer register from aligned memory using a mask representing 32-bit integers.
 *
 * @tparam T1 Pointer type (must be `__m256i*`).
 * @tparam T2 Mask type (`__m256i`).
 * @param pointer Pointer to load the value from.
 * @param mask Mask to apply to the load operation.
 */
template<typename T1, typename T2>
constexpr auto LoadMaskInt32(T1 pointer, T2 mask) { return _mm256_maskload_epi32(pointer, mask); }

/** Loads a 256-bit integer register from aligned memory using a mask representing 64-bit integers.
 *
 * @tparam T1 Pointer type (must be `__m256i*`).
 * @tparam T2 Mask type (`__m256i`).
 * @param pointer Pointer to load the value from.
 * @param mask Mask to apply to the load operation.
 */
template<typename T1, typename T2>
constexpr auto LoadMaskInt64(T1 pointer, T2 mask) { return _mm256_maskload_epi64(pointer, mask); }

/** Stores a 256-bit floating-point register into aligned memory.
 *
 * @tparam T1 Pointer type (must be `__m256*`).
 * @tparam T2 Value type (`__m256`).
 * @param pointer Pointer to store the value.
 * @param value Value to store.
 */
template<typename T1, typename T2>
constexpr auto StoreFloat(T1 pointer, T2  value) { return _mm256_store_ps(pointer, value); }

/** Stores a 256-bit floating-point register into aligned memory using a mask.
 *
 * @tparam T1 Pointer type (must be `__m256*`).
 * @tparam T2 Mask type (`__m256`).
 * @tparam T3 Value type (`__m256`).
 * @param pointer Pointer to store the value.
 * @param mask Mask to apply to the store operation.
 * @param value Value to store.
 */
template<typename T1, typename T2, typename T3>
constexpr auto StoreMaskFloat(T1 pointer, T2  mask, T3 value) { return _mm256_maskstore_ps(pointer, mask, value); }

/** Loads a 256-bit floating-point register from aligned memory.
 *
 * @tparam T Pointer type (must be `__m256*`).
 * @param pointer Pointer to load the value from.
 */
template<typename T>
constexpr auto LoadFloat(T pointer) { return _mm256_load_ps(pointer); }

/** Loads a 256-bit floating-point register from aligned memory using a mask.
 *
 * @tparam T1 Pointer type (must be `__m256*`).
 * @tparam T2 Mask type (`__m256`).
 * @param pointer Pointer to load the value from.
 * @param mask Mask to apply to the load operation.
 */
template<typename T1, typename T2>
constexpr auto LoadMaskFloat(T1 pointer, T2 mask) { return _mm256_maskload_ps(pointer, mask); }

/** Stores a 256-bit double-precision floating-point register into aligned memory.
 *
 * @tparam T1 Pointer type (must be `__m256d*`).
 * @tparam T2 Value type (`__m256d`).
 * @param pointer Pointer to store the value.
 * @param value Value to store.
 */
template<typename T1, typename T2>
constexpr auto StoreDouble(T1 pointer, T2  value) { return _mm256_store_pd(pointer, value); }

/** Stores a 256-bit double-precision floating-point register into aligned memory using a mask.
 *
 * @tparam T1 Pointer type (must be `__m256d*`).
 * @tparam T2 Mask type (`__m256d`).
 * @tparam T3 Value type (`__m256d`).
 * @param pointer Pointer to store the value.
 * @param mask Mask to apply to the store operation.
 * @param value Value to store.
 */
template<typename T1, typename T2, typename T3>
constexpr auto StoreMaskDouble(T1 pointer, T2  mask, T3 value) { return _mm256_maskstore_pd(pointer, mask, value); }

/** Loads a 256-bit double-precision floating-point register from aligned memory.
 *
 * @tparam T Pointer type (must be `__m256d*`).
 * @param pointer Pointer to load the value from.
 */
template<typename T>
constexpr auto LoadDouble(T pointer) { return _mm256_load_pd(pointer); }

/** Loads a 256-bit double-precision floating-point register from aligned memory using a mask.
 *
 * @tparam T1 Pointer type (must be `__m256d*`).
 * @tparam T2 Mask type (`__m256d`).
 * @param pointer Pointer to load the value from.
 * @param mask Mask to apply to the load operation.
 */
template<typename T1, typename T2>
constexpr auto LoadMaskDouble(T1 pointer, T2 mask) { return _mm256_maskload_pd(pointer, mask); }

/** Adds two 256-bit integer registers representing 32 8-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto AddInt8(T1 pointer1, T2  pointer2) { return _mm256_add_epi8(pointer1, pointer2); }

/** Adds two 256-bit integer registers representing 16 16-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto AddInt16(T1 pointer1, T2  pointer2) { return _mm256_add_epi16(pointer1, pointer2); }

/** Adds two 256-bit integer registers representing 8 32-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto AddInt32(T1 pointer1, T2  pointer2) { return _mm256_add_epi32(pointer1, pointer2); }

/** Adds two 256-bit integer registers representing 4 64-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto AddInt64(T1 pointer1, T2  pointer2) { return _mm256_add_epi64(pointer1, pointer2); }

/** Adds two 256-bit floating-point registers.
 *
 * @tparam T1 First register type (`__m256`).
 * @tparam T2 Second register type (`__m256`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto AddFloat(T1 pointer1, T2  pointer2) { return _mm256_add_ps(pointer1, pointer2); }

/** Adds two 256-bit double-precision floating-point registers.
 *
 * @tparam T1 First register type (`__m256d`).
 * @tparam T2 Second register type (`__m256d`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto AddDouble(T1 pointer1, T2  pointer2) { return _mm256_add_pd(pointer1, pointer2); }

/** Subtracts two 256-bit integer registers representing 32 8-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto SubInt8(T1 pointer1, T2  pointer2) { return _mm256_sub_epi8(pointer1, pointer2); }

/** Subtracts two 256-bit integer registers representing 16 16-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto SubInt16(T1 pointer1, T2  pointer2) { return _mm256_sub_epi16(pointer1, pointer2); }

/** Subtracts two 256-bit integer registers representing 8 32-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto SubInt32(T1 pointer1, T2  pointer2) { return _mm256_sub_epi32(pointer1, pointer2); }

/** Subtracts two 256-bit integer registers representing 4 64-bit integers.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto SubInt64(T1 pointer1, T2  pointer2) { return _mm256_sub_epi64(pointer1, pointer2); }

/** Subtracts two 256-bit floating-point registers.
 *
 * @tparam T1 First register type (`__m256`).
 * @tparam T2 Second register type (`__m256`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto SubFloat(T1 pointer1, T2  pointer2) { return _mm256_sub_ps(pointer1, pointer2); }

/** Subtracts two 256-bit double-precision floating-point registers.
 *
 * @tparam T1 First register type (`__m256d`).
 * @tparam T2 Second register type (`__m256d`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto SubDouble(T1 pointer1, T2  pointer2) { return _mm256_sub_pd(pointer1, pointer2); }

/** Multiplies two 256-bit float registers elementwise, where each element is a floating-point.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto MulFloat(T1 pointer1, T2  pointer2) { return _mm256_mul_ps(pointer1, pointer2); }

/** Multiplies two 256-bit double registers elementwise, where each element is a double-precision floating-point.
 *
 * @tparam T1 First register type (`__m256i`).
 * @tparam T2 Second register type (`__m256i`).
 * @param pointer1 First register.
 * @param pointer2 Second register.
 */
template<typename T1, typename T2>
constexpr auto MulDouble(T1 pointer1, T2  pointer2) { return _mm256_mul_pd(pointer1, pointer2); }

/** Performs fused multiply-add operation for 256-bit floating-point values.
 *
 * @param value1 First operand.
 * @param value2 Second operand.
 * @param value3 Third operand.
 * @return Result of (value1 * value2) + value3.
 */
template<typename T1, typename T2, typename T3>
constexpr auto FMAFloat(T1 value1, T2  value2, T3 value3) { return _mm256_fmadd_ps(value1, value2, value3); }

/** Performs fused multiply-add operation for 256-bit double-precision floating-point values.
 *
 * @param value1 First operand.
 * @param value2 Second operand.
 * @param value3 Third operand.
 * @return Result of (value1 * value2) + value3.
 */
template<typename T1, typename T2, typename T3>
constexpr auto FMADouble(T1 value1, T2  value2, T3 value3) { return _mm256_fmadd_pd(value1, value2, value3); }

/** Broadcasts a 64-bit scalar value to all elements of a 256-bit integer register.
 *
 * @tparam T1 Scalar value type.
 * @param value Scalar value to broadcast.
 * @return A 256-bit integer register with all elements set to the scalar value.
 */
template<typename T1>
constexpr auto BroadcastInt64(T1 value) { return _mm256_set1_epi64x(value); }

/** Broadcasts a 32-bit scalar value to all elements of a 256-bit integer register.
 *
 * @tparam T1 Scalar value type.
 * @param value Scalar value to broadcast.
 * @return A 256-bit integer register with all elements set to the scalar value.
 */
template<typename T1>
constexpr auto BroadcastInt32(T1 value) { return _mm256_set1_epi32(value); }

/** Broadcasts a 16-bit scalar value to all elements of a 256-bit integer register.
 *
 * @tparam T1 Scalar value type.
 * @param value Scalar value to broadcast.
 * @return A 256-bit integer register with all elements set to the scalar value.
 */
template<typename T1>
constexpr auto BroadcastInt16(T1 value) { return _mm256_set1_epi16(value); }

/** Broadcasts an 8-bit scalar value to all elements of a 256-bit integer register.
 *
 * @tparam T1 Scalar value type.
 * @param value Scalar value to broadcast.
 * @return A 256-bit integer register with all elements set to the scalar value.
 */
template<typename T1>
constexpr auto BroadcastInt8(T1 value) { return _mm256_set1_epi8(value); }

/** Broadcasts a floating-point value to all elements of a 256-bit floating-point register.
 *
 * @tparam T1 Scalar value type.
 * @param value Scalar value to broadcast.
 * @return A 256-bit floating-point register with all elements set to the scalar value.
 */
template<typename T1>
constexpr auto BroadcastFloat(T1 value) { return _mm256_set1_ps(value); }

/** Broadcasts a double-precision value to all elements of a 256-bit double-precision floating-point register.
 *
 * @tparam T1 Scalar value type.
 * @param value Scalar value to broadcast.
 * @return A 256-bit double-precision floating-point register with all elements set to the scalar value.
 */
template<typename T1>
constexpr auto BroadcastDouble(T1 value) { return _mm256_set1_pd(value); }

/** Reduces a 256-bit floating-point register to a single float using horizontal addition.
 *
 * @param value The register to reduce.
 * @return The final reduced float.
 */
template<typename T>
auto Unroll32(T value) {
	const __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
	const __m128 r2 = _mm_add_ps(r1, _mm_movehl_ps(r1, r1));
	const __m128 r3 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
	return _mm_cvtss_f32(r3);
}

/** Reduces a 256-bit double-precision floating-point register to a single double using horizontal addition.
 *
 * @param value The register to reduce.
 * @return The final reduced double.
 */
template<typename T>
auto Unroll64(T value) {
	const __m128d r1 = _mm256_extractf128_pd(value, 1);
	const __m128d r2 = _mm_add_pd(r1, _mm256_castpd256_pd128(value));
	
	__m128 undef = _mm_undefined_ps();
	__m128 r3 = _mm_movehl_ps(undef, _mm_castpd_ps(r2));
	__m128d r4 = _mm_castps_pd(r3);
	
	return  _mm_cvtsd_f64(_mm_add_sd(r2, r4));
}

/**
 * @brief Generates a mask for selecting elements in a 256-bit integer register based on size.
 *
 * @details
 * This function creates a mask for use in masked AVX2 operations. It generates a 256-bit integer mask
 * with `size` elements set to `-1` (active) and the remaining set to `1` (inactive).
 *
 * @tparam size The number of active elements in the mask (between 1 and 8).
 * @return A 256-bit integer mask for conditional operations.
 */
template<int size>
inline auto MaskBySize32() { 
	if constexpr (size == 1) return _mm256_setr_epi32(-1, 1, 1, 1, 1, 1, 1, 1);
	else if constexpr (size == 2) return _mm256_setr_epi32(-1, -1, 1, 1, 1, 1, 1, 1);
	else if constexpr (size == 3) return _mm256_setr_epi32(-1, -1, -1, 1, 1, 1, 1, 1); 
	else if constexpr (size == 4) return _mm256_setr_epi32(-1, -1, -1, -1, 1, 1, 1, 1);
	else if constexpr (size == 5) return _mm256_setr_epi32(-1, -1, -1, -1, -1, 1, 1, 1);
	else if constexpr (size == 6) return _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 1, 1);
	else if constexpr (size == 7) return _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 1);
	else return _mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1);
}

/**
 * @brief Generates a mask for selecting elements in a 256-bit integer register based on size.
 *
 * @details
 * This function creates a mask for use in masked AVX2 operations. It generates a 256-bit integer mask
 * with `size` elements set to `-1` (active) and the remaining set to `1` (inactive).
 *
 * @tparam size The number of active elements in the mask (between 1 and 4).
 * @return A 256-bit integer mask for conditional operations.
 */
template<int size>
inline auto MaskBySize64() {
	if constexpr (size == 1) return _mm256_setr_epi64x(-1, 1, 1, 1);
	else if constexpr (size == 2) return _mm256_setr_epi64x(-1, -1, 1, 1);
	else if constexpr (size == 3) return _mm256_setr_epi64x(-1, -1, -1, 1);
	else return _mm256_setr_epi64x(1, 1, 1, 1);
}
