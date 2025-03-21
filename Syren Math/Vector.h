#pragma once

#ifdef USING_AVX2
#include "AVX2.h"
#endif

#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cstddef> 

#include <cassert>
#include <stdexcept>

#include "Aligned Allocator.h"

#define assertm(exp, msg) assert(((void)msg, exp))


namespace SyrenMath {
	/** Template structure to check if a common type exists for a given parameter pack.
	 * 
	 * @details 
	 * This structure is used to determine if a common type can be deduced from a provided
	 * parameter pack `Ts...`. It specializes to `std::true_type` if the common type exists,
	 * otherwise remains as `std::false_type`.
	 *
	 * @tparam AlwaysVoid A type placeholder. Not used in the specialization.
	 * @tparam Ts Parameter pack of types to check for common type.
	 */
	template <typename AlwaysVoid, typename... Ts>
	struct has_common_type_impl : std::false_type {};

	/** Specialized structure for checking the existence of a common type.
	 *
	 * @details 
	 * This specialization utilizes `std::void_t` and `std::common_type_t` to deduce if
	 * a common type exists for the types in the parameter pack `Ts...`.
	 *
	 * @tparam Ts Parameter pack of types to check for common type.
	 */
	template <typename... Ts>
	struct has_common_type_impl<std::void_t<std::common_type_t<Ts...>>, Ts...> : std::true_type {};

	/** Concept to check for common type in a parameter pack.
	 * 
	 * @details 
	 * This concept evaluates to true if the number of types in `Ts...` is less than 2,
	 * or if a common type exists among the provided types. It is useful for generalizing
	 * functions that can accept multiple types while ensuring type safety.
	 *
	 * @tparam Ts Parameter pack of types to evaluate for common type.
	 */
	template <typename... Ts>
	concept has_common_type =
		sizeof...(Ts) < 2 ||
		has_common_type_impl<void, Ts...>::value;

	/** A fixed-size mathematical vector class template.
	 * 
	 * @details 
	 * This class template provides a type-safe, fixed-size vector similar to std::vector,
	 * but specifically designed for arithmetic types only. It supports initialization,
	 * element access, and element-wise arithmetic operations using SIMD instructions for various numeric types.
	 *
	 * @tparam T The type of elements in the vector. Must be an arithmetic type.
	 * @tparam size The number of elements in the vector.
	 */
	template<typename T, unsigned int size>
	// TODO: Add multithreading support for SIMD operations.
	class Vector {
	private:
		static_assert(std::is_arithmetic<T>::value, "Not an arithmetic type.");		
		
		/** Type alias for an aligned vector with a specified alignment in bytes.
		 * 
		 * @details
		 * This alias uses an aligned allocator to ensure proper memory alignment.
		 */
		template <typename U, std::size_t ALIGNMENT_IN_BYTES = 64>
		using AlignedVector = std::vector<U, SyrenUtility::AlignedAllocator<U, ALIGNMENT_IN_BYTES> >;
		
		AlignedVector<T, 64> mData; /*!< The underlying data storage for the vector. */
		bool mIsColumn;
	public:
		/** Default constructor for the vector.
		 * 
		 * @details
		 * Initializes all elements to zero.
		 */
		Vector() : mData(size, static_cast<T>(0)), mIsColumn(true) { }

		/** Constructor that fills the vector with a specified value.
		 * 
		 * @tparam U Type of the fill value. Must be an arithmetic type.
		 * @param pFill The value to initialize all elements of the vector.
		 */
		template<typename U>
		Vector(U pFill) : mData(size, static_cast<T>(0)), mIsColumn(true) {
			static_assert(std::is_arithmetic<U>::value, "Not an arithmetic type.");
			std::fill(mData.begin(), mData.end(), static_cast<T>(pFill));
		}
		
		/** Constructor that initializes the vector with provided values.
		 * 
		 * @details
		 * This constructor takes a parameter pack of values to populate the vector.
		 * The number of provided values must match the size of the vector.
		 *
		 * @tparam Ts Parameter pack of values to initialize the vector.
		 * @param pData Values used to initialize the vector.
		 */
		template<typename... Ts> requires has_common_type<Ts...>
		Vector(const Ts&... pData) {
			assertm(sizeof...(Ts) == size, "Number of arguments provided does not match the size of the vector.");
			mData = { static_cast<T>(pData)... };
			mIsColumn = true;
		}

		/** Constructs the vector where the other vector is of a different arithmetic type.
		 *
		 * @details
		 * This constructor allows implicit type conversion from `Vector<U, size>` to `Vector<T, size>`.
		 * It ensures type safety by requiring `U` to be an arithmetic type and uses `std::transform`
		 * to convert elements from `rhs` into the correct type `T`.
		 *
		 * @tparam U The type of elements in the source vector.
		 * @param rhs The vector to copy elements from.
		 */
		template<typename U>
		Vector(Vector<U, size>& rhs) : mData(size, static_cast<T>(0)), mIsColumn(true) {
			static_assert(std::is_arithmetic<U>::value, "Not an arithmetic type.");

			std::transform(rhs.begin(), rhs.end(), this->begin(), [](U x) { return (T)x; });
			mIsColumn = rhs.isColumnVector();
		}

		/** Copy constructor for the vector where the other vector is of a different arithmetic type.
		 *
		 * @details
		 * This constructor allows implicit type conversion from `Vector<U, size>` to `Vector<T, size>`.
		 * It ensures type safety by requiring `U` to be an arithmetic type and uses `std::transform`
		 * to convert elements from `rhs` into the correct type `T`.
		 *
		 * @tparam U The type of elements in the source vector.
		 * @param rhs The vector to copy elements from.
		 */
		template<typename U>
		Vector(const Vector<U, size>& rhs) : mData(size, static_cast<T>(0)), mIsColumn(true) {
			static_assert(std::is_arithmetic<U>::value, "Not an arithmetic type.");

			std::transform(rhs.begin(), rhs.end(), this->begin(), [](U x) { return (T)x; });
			mIsColumn = rhs.isColumnVector();
		}

		/** Copy constructor for the vector.
		 *
		 * @details
		 * Initializes the vector with the values from another vector of the same size.
		 *
		 * @param rhs The vector to copy values from.
		 */
		Vector(const Vector<T, size>& rhs) : mData(rhs.mData), mIsColumn(rhs.mIsColumn) {}

		/** Move constructor for the vector.
		 *
		 * @details
		 * Moves the values from another vector of the same size into this vector.
		 *
		 * @param other The vector to move values from.
		 */
		Vector(Vector&& other) noexcept : mData(std::move(other.mData)), mIsColumn(other.mIsColumn) {}

		/** Accesses an element of the vector by index. 
		 *
		 * @param index The index of the element to access.
		 * @return Reference to the element at the specified index.
		 * @throws std::out_of_range if the index is out of bounds.
		 */
		T& operator[](unsigned int index)
		{
			assert(index < size && "index out of range");
			if(index < mData.size()) return mData[index];
			throw std::out_of_range("index out of range");
		}

		/** Accesses an element of the vector by index (const).
		 *
		 * @param index The index of the element to access.
		 * @return Const reference to the element at the specified index.
		 * @throws std::out_of_range if the index is out of bounds.
		 */
		const T& operator[](unsigned int index) const
		{
			assert(index < size && "index out of range");
			if (index < mData.size()) return mData[index];
			throw std::out_of_range("index out of range");
		}

		/** Move assignment operator for the vector.
		 *
		 * @details
		 * Copies the values from another vector of the same size into this vector.
		 *
		 * @param rhs The vector to copy values from.
		 * @return Reference to this vector after assignment.
		 */
		Vector& operator=(Vector&& other) noexcept {
			if (this != &other) {
				mData = std::move(other.mData);
				mIsColumn = other.mIsColumn;
			}
			return *this;
		}
		
		/** Assignment operator for the vector.
		 *
		 * @details
		 * Copies the values from another vector of the same size into this vector.
		 *
		 * @param rhs The vector to copy values from.
		 * @return Reference to this vector after assignment.
		 */
		template<typename U>
		Vector<T, size>& operator=(Vector<U, size>& rhs) {
			static_assert(std::is_arithmetic<U>::value, "Not an arithmetic type.");
			
			std::transform(rhs.begin(), rhs.end(), this->begin(), [](U x) { return (T)x; });
			mIsColumn = rhs.isColumnVector();

			return *this;
		}

		constexpr bool isColumnVector() const { return mIsColumn; } /*!< Returns true if the vector is a column vector. */
		auto length() { return mData.size() ; } /*!< Returns the number of elements in the vector. */
		auto begin() { return mData.begin() ; } /*!< Returns an iterator to the beginning of the vector. */
		auto begin() const { return mData.begin(); } /*!< Returns an iterator to the beginning of the vector (const). */
		auto end() { return mData.end() ; } /*!< Returns an iterator to the end of the vector. */
		auto end() const { return mData.end(); } /*!< Returns an iterator to the end of the vector (const). */
	public:
		/** Element-wise addition for two vectors of type __int64.
		 * 
		 * @details
		 * This operator overload performs element-wise addition for two vectors of type
		 * __int64 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to add to this vector.
		 * @return A new Vector<__int64, size> containing the sum of the two vectors.
		 */
		Vector<__int64, size> operator+(Vector<__int64, size> const& rhs) {
			static_assert(std::is_same<T, __int64>::value, "Vector types do not match for 64 bit addition.");
			Vector<__int64, size> sum(0);
			const int alignedElements = size - size % CMAXINT64;

			for (int i = 0; i < alignedElements; i += CMAXINT64) {
				StoreInt((SIMDInt*) & (sum)[i],
					AddInt64(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			auto mask = MaskBySize64<size % CMAXINT64>();
			StoreMaskInt64(&(sum)[alignedElements], mask,
				AddInt(
					LoadMaskInt64(&mData[alignedElements], mask),
					LoadMaskInt64(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}
		
		/** Element-wise addition for two vectors of type __int32.
		 *
		 * @details
		 * This operator overload performs element-wise addition for two vectors of type
		 * __int32 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to add to this vector.
		 * @return A new Vector<__int32, size> containing the sum of the two vectors.
		 */
		Vector<__int32, size> operator+(Vector<__int32, size> const& rhs) {
			static_assert(std::is_same<T, __int32>::value, "Vector types do not match for 32 bit addition.");
			Vector<__int32, size> sum(0);
			const int alignedElements = size - size % CMAXINT32;

			for (int i = 0; i < alignedElements; i += CMAXINT32) {
				StoreInt((SIMDInt*) & (sum)[i],
					AddInt32(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			auto mask = MaskBySize32<size % CMAXINT32>();
			StoreMaskInt32(&(sum)[alignedElements], mask,
				AddInt32(
					LoadMaskInt32(&mData[alignedElements], mask),
					LoadMaskInt32(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Element-wise addition for two vectors of type __int16.
		 * 
		 * @details
		 * This operator overload performs element-wise addition for two vectors of type
		 * __int16 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to add to this vector.
		 * @return A new Vector<__int16, size> containing the sum of the two vectors.
		 */
		Vector<__int16, size> operator+(Vector<__int16, size> const& rhs) {
			static_assert(std::is_same<T, __int16>::value, "Vector types do not match for 16 bit addition.");
			Vector<__int16, size> sum(0);
			const int alignedElements = size - size % CMAXINT16;

			for (int i = 0; i < alignedElements; i += CMAXINT16) {
				StoreInt((SIMDInt*) & (sum)[i],
					AddInt16(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			if (alignedElements != size) {
				for (int i = alignedElements; i < size; ++i) {
					(sum)[i] = mData[i] + rhs[i];
				}
			}

			return(sum);
		}

		/** Element-wise addition for two vectors of type __int8.
		 * 
		 * @details
		 * This operator overload performs element-wise addition for two vectors of type
		 * __int8 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to add to this vector.
		 * @return A new Vector<__int8, size> containing the sum of the two vectors.
		 */
		Vector<__int8, size> operator+(Vector<__int8, size> const& rhs) {
			static_assert(std::is_same<T, __int8>::value, "Vector types do not match for 8 bit addition.");
			Vector<__int8, size> sum(0);
			const int alignedElements = size - size % CMAXINT8;

			for (int i = 0; i < alignedElements; i += CMAXINT8) {
				StoreInt((SIMDInt*) & (sum)[i],
					AddInt8(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			if (alignedElements != size) {
				for (int i = alignedElements; i < size; ++i) {
					(sum)[i] = mData[i] + rhs[i];
				}
			}

			return(sum);
		}

		/** Element-wise addition for two vectors of type float.
		 *  
		 * @details
		 * This operator overload performs element-wise addition for two vectors of type
		 * float and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to add to this vector.
		 * @return A new Vector<float, size> containing the sum of the two vectors.
		 */
		Vector<float, size> operator+(Vector<float, size> const& rhs) {
			static_assert(std::is_same<T, float>::value, "Vector types do not match for float addition.");
			Vector<float, size> sum(0);
			const int alignedElements = size - size % CMAXFLOAT;

			for (int i = 0; i < alignedElements; i += CMAXFLOAT) {
				StoreFloat(& (sum)[i],
					AddFloat(LoadFloat(& mData[i]), LoadFloat(& rhs[i]))
				);
			}

			auto mask = MaskBySize32<size % CMAXFLOAT>();
			StoreMaskFloat(&(sum)[alignedElements], mask,
				AddFloat(
					LoadMaskFloat(&mData[alignedElements], mask),
					LoadMaskFloat(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Element-wise addition for two vectors of type double.
		 *  
		 * @details
		 * This operator overload performs element-wise addition for two vectors of type
		 * double and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to add to this vector.
		 * @return A new Vector<double, size> containing the sum of the two vectors.
		 */
		Vector<double, size> operator+(Vector<double, size> const& rhs) {
			static_assert(std::is_same<T, double>::value, "Vector types do not match for double addition.");
			Vector<double, size> sum(0);
			const int alignedElements = size - size % CMAXDOUBLE;

			for (int i = 0; i < alignedElements; i += CMAXDOUBLE) {
				StoreDouble(& (sum)[i],
					AddDouble(LoadDouble(& mData[i]), LoadDouble(& rhs[i]))
				);
			}

			auto mask = MaskBySize64<size % CMAXDOUBLE>();
			StoreMaskDouble(&(sum)[alignedElements], mask,
				AddDouble(
					LoadMaskDouble(&mData[alignedElements], mask),
					LoadMaskDouble(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Element-wise subtraction for two vectors of type __int64.
		 *
		 * @details
		 * This operator overload performs element-wise subtraction for two vectors of type
		 * __int64 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to subtract from this vector.
		 * @return A new Vector<__int64, size> containing the difference of the two vectors.
		 */
		Vector<__int64, size> operator-(Vector<__int64, size> const& rhs) {
			static_assert(std::is_same<T, __int64>::value, "Vector types do not match for 64 bit subtraction.");
			Vector<__int64, size> sum(0);
			const int alignedElements = size - size % CMAXINT64;

			for (int i = 0; i < alignedElements; i += CMAXINT64) {
				StoreInt((SIMDInt*) & (sum)[i],
					SubInt64(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			auto mask = MaskBySize64<size % CMAXINT64>();
			StoreMaskInt64(&(sum)[alignedElements], mask,
				SubInt(
					LoadMaskInt64(&mData[alignedElements], mask),
					LoadMaskInt64(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Element-wise subtraction for two vectors of type __int32.
		 *
		 * @details
		 * This operator overload performs element-wise subtraction for two vectors of type
		 * __int32 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to subtract from this vector.
		 * @return A new Vector<__int32, size> containing the difference of the two vectors.
		 */
		Vector<__int32, size> operator-(Vector<__int32, size> const& rhs) {
			static_assert(std::is_same<T, __int32>::value, "Vector types do not match for 32 bit subtraction.");
			Vector<__int32, size> sum(0);
			const int alignedElements = size - size % CMAXINT32;

			for (int i = 0; i < alignedElements; i += CMAXINT32) {
				StoreInt((SIMDInt*) & (sum)[i],
					SubInt32(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			auto mask = MaskBySize32<size % CMAXINT32>();
			StoreMaskInt32(&(sum)[alignedElements], mask,
				SubInt(
					LoadMaskInt32(&mData[alignedElements], mask),
					LoadMaskInt32(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Element-wise subtraction for two vectors of type __int16.
		 *
		 * @details
		 * This operator overload performs element-wise subtraction for two vectors of type
		 * __int16 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to subtract from this vector.
		 * @return A new Vector<__int16, size> containing the difference of the two vectors.
		*/
		Vector<__int16, size> operator-(Vector<__int16, size> const& rhs) {
			static_assert(std::is_same<T, __int16>::value, "Vector types do not match for 16 bit subtraction.");
			Vector<__int16, size> sum(0);
			const int alignedElements = size - size % CMAXINT16;

			for (int i = 0; i < alignedElements; i += CMAXINT16) {
				StoreInt((SIMDInt*) & (sum)[i],
					SubInt16(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			if (alignedElements != size) {
				for (int i = alignedElements; i < size; ++i) {
					(sum)[i] = mData[i] - rhs[i];
				}
			}

			return(sum);
		}

		/** Element-wise subtraction for two vectors of type __int8.
		 *
		 * @details
		 * This operator overload performs element-wise subtraction for two vectors of type
		 * __int8 and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to subtract from this vector.
		 * @return A new Vector<__int8, size> containing the difference of the two vectors.
		 */
		Vector<__int8, size> operator-(Vector<__int8, size> const& rhs) {
			static_assert(std::is_same<T, __int8>::value, "Vector types do not match for 8 bit subtraction.");
			Vector<__int8, size> sum(0);
			const int alignedElements = size - size % CMAXINT8;

			for (int i = 0; i < alignedElements; i += CMAXINT8) {
				StoreInt((SIMDInt*) & (sum)[i],
					SubInt8(LoadInt((SIMDInt*) & mData[i]), LoadInt((SIMDInt*) & rhs[i]))
				);
			}

			if (alignedElements != size) {
				for (int i = alignedElements; i < size; ++i) {
					(sum)[i] = mData[i] - rhs[i];
				}
			}

			return(sum);
		}

		/** Element-wise subtraction for two vectors of type float.
		 *
		 * @details
		 * This operator overload performs element-wise subtraction for two vectors of type
		 * float and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to subtract from this vector.
		 * @return A new Vector<float, size> containing the difference of the two vectors.
		 */
		Vector<float, size> operator-(Vector<float, size> const& rhs) {
			static_assert(std::is_same<T, float>::value, "Vector types do not match for float subtraction.");
			Vector<float, size> sum(0);
			const int alignedElements = size - size % CMAXFLOAT;

			for (int i = 0; i < alignedElements; i += CMAXFLOAT) {
				StoreFloat(&(sum)[i],
					SubFloat(LoadFloat(&mData[i]), LoadFloat(&rhs[i]))
				);
			}

			auto mask = MaskBySize32<size % CMAXFLOAT>();
			StoreMaskFloat(&(sum)[alignedElements], mask,
				SubFloat(
					LoadMaskFloat(&mData[alignedElements], mask),
					LoadMaskFloat(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Element-wise subtraction for two vectors of type double.
		 *
		 * @details
		 * This operator overload performs element-wise subtraction for two vectors of type
		 * double and returns a new vector containing the results. It uses SIMD instructions
		 * for optimized performance.
		 *
		 * @param rhs The right-hand side vector to subtract from this vector.
		 * @return A new Vector<double, size> containing the difference of the two vectors.
		 */
		Vector<double, size> operator-(Vector<double, size> const& rhs) {
			static_assert(std::is_same<T, double>::value, "Vector types do not match for double subtraction.");
			Vector<double, size> sum(0);
			const int alignedElements = size - size % CMAXDOUBLE;

			for (int i = 0; i < alignedElements; i += CMAXDOUBLE) {
				StoreDouble(&(sum)[i],
					SubDouble(LoadDouble(&mData[i]), LoadDouble(&rhs[i]))
				);
			}

			auto mask = MaskBySize64<size % CMAXDOUBLE>();
			StoreMaskDouble(&(sum)[alignedElements], mask,
				SubDouble(
					LoadMaskDouble(&mData[alignedElements], mask),
					LoadMaskDouble(&rhs[alignedElements], mask)
				)
			);

			return(sum);
		}

		/** Transposes a column vector into a row vector.
		 *
		 * @details
		 * This function modifies the column vector into a row vector, preserving its data.
		 * It returns a copy of the transposed vector.
		 *
		 * @return A new `Vector<T, size>` that represents the row vector equivalent.
		 */
		Vector<T, size> t() {
			this->mIsColumn = false;
			Vector<T, size> returnValue = *this;
			this->mIsColumn = true;
			return returnValue;
		}

		/** Performs dot product multiplication of two 64-bit integer vectors.
		 *
		 * @details
		 * This function ensures that multiplication is only performed between a row vector and a column vector.
		 * It casts the integer vectors to floating-point for computation and then converts the result back to an integer.
		 *
		 * @param rhs The right-hand side vector.
		 * @return The dot product result as a 64-bit integer.
		 * @throws std::runtime_error If the vectors are not in the correct row-column multiplication format.
		 */
		__int64 operator*(Vector<__int64, size>& rhs) {
			static_assert(std::is_same<T, __int64>::value, "Vector types do not match for 64 bit vector multiplication.");
			if (this->isColumnVector() || !rhs.isColumnVector()) throw std::runtime_error("Only a row vector and a column vector can be multiplied.");

			Vector<double, size> lhsF = *this;
			Vector<double, size> rhsF = rhs;

			return((__int64)(lhsF * rhsF));
		}

		/** Performs dot product multiplication of two 32-bit integer vectors.
		 *
		 * @details
		 * This function ensures that multiplication is only performed between a row vector and a column vector.
		 * It casts the integer vectors to floating-point for computation and then converts the result back to an integer.
		 *
		 * @param rhs The right-hand side vector.
		 * @return The dot product result as a 32-bit integer.
		 * @throws std::runtime_error If the vectors are not in the correct row-column multiplication format.
		 */
		__int32 operator*(Vector<__int32, size>& rhs) {
			static_assert(std::is_same<T, __int32>::value, "Vector types do not match for 32 bit vector multiplication.");
			if (this->isColumnVector() || !rhs.isColumnVector()) throw std::runtime_error("Only a row vector and a column vector can be multiplied.");

			Vector<float, size> lhsF = *this;
			Vector<float, size> rhsF = rhs;

			return((__int32)(lhsF * rhsF));
		}

		/** Performs dot product multiplication of two 16-bit integer vectors.
		 *
		 * @details
		 * This function ensures that multiplication is only performed between a row vector and a column vector.
		 * It casts the integer vectors to floating-point for computation and then converts the result back to an integer.
		 *
		 * @param rhs The right-hand side vector.
		 * @return The dot product result as a 16-bit integer.
		 * @throws std::runtime_error If the vectors are not in the correct row-column multiplication format.
		 */
		float operator*(Vector<float, size> const& rhs) {
			static_assert(std::is_same<T, float>::value, "Vector types do not match for float vector multiplication.");
			if (this->isColumnVector() || !rhs.isColumnVector()) throw std::runtime_error("Only a row vector and a column vector can be multiplied.");

			Vector<float, CMAXFLOAT> result(0);
			const int alignedElements = size - size % CMAXFLOAT;

			for (int i = 0; i < alignedElements; i += CMAXFLOAT) {
				StoreFloat(&(result)[0],
					FMAFloat(LoadFloat(&mData[i]), LoadFloat(&rhs[i]), LoadFloat(&result[0]))
				);
			}

			auto mask = MaskBySize32<size % CMAXFLOAT>();
			StoreFloat(&(result)[0],
				FMAFloat(
					LoadMaskFloat(&mData[alignedElements], mask),
					LoadMaskFloat(&rhs[alignedElements], mask),
					LoadFloat(&result[0])
				)
			);

			return(Unroll32(LoadFloat(&result[0])));
		}

		/** Performs dot product multiplication of two 64-bit integer vectors.
		 *
		 * @details
		 * This function ensures that multiplication is only performed between a row vector and a column vector.
		 * It casts the integer vectors to floating-point for computation and then converts the result back to an integer.
		 *
		 * @param rhs The right-hand side vector.
		 * @return The dot product result as a 64-bit integer.
		 * @throws std::runtime_error If the vectors are not in the correct row-column multiplication format.
		 */
		double operator*(Vector<double, size> const& rhs) {
			static_assert(std::is_same<T, double>::value, "Vector types do not match for double vector multiplication.");
			if (this->isColumnVector() || !rhs.isColumnVector()) throw std::runtime_error("Only a row vector and a column vector can be multiplied.");

			Vector<double, CMAXDOUBLE> result(0);
			const int alignedElements = size - size % CMAXDOUBLE;

			for (int i = 0; i < alignedElements; i += CMAXDOUBLE) {
				StoreDouble(&(result)[0],
					FMADouble(LoadDouble(&mData[i]), LoadDouble(&rhs[i]), LoadDouble(&result[0]))
				);
			}

			auto mask = MaskBySize64<size % CMAXDOUBLE>();
			StoreDouble(&(result)[0],
				FMADouble(
					LoadMaskDouble(&mData[alignedElements], mask),
					LoadMaskDouble(&rhs[alignedElements], mask),
					LoadDouble(&result[0])
				)
			);

			return(Unroll64(LoadDouble(&result[0])));
		}

		/** Performs scalar multiplication of a vector by a scalar value.
		 *
		 * @details
		 * This function multiplies each element in the vector by a given scalar and returns a new vector with the results.
		 * It uses SIMD instructions for optimized performance.
		 *
		 * @tparam U The type of the scalar value. Must be a floating-point or integer type.
		 * @param scalar The scalar value to multiply.
		 * @return A new Vector<T, size> containing the elements of the original vector multiplied by the scalar.
		 * @throws std::runtime_error If the scalar type is not compatible with the vector type.
		 */
		template <typename U, typename std::enable_if<
			std::is_same<float, U>::value ||
			std::is_same<__int32, U>::value ||
			std::is_same<__int64, U>::value ||
			std::is_same<double, U>::value, int>::type = 0>
		Vector<T, size> operator*(const U scalar) {
			static_assert(!((std::is_same<U, double>::value || std::is_same<U, __int64>::value) &&
				(std::is_same<T, float>::value || std::is_same<T, __int32>::value)),
				"Scalar multiplication is only defined when there is no potencial loss of data. Select a scalar with the same or fewer number of bits.");

			if constexpr ((std::is_same<T, double>::value || std::is_same<T, __int64>::value)) {
				Vector<double, size> scaled(0);
				Vector<double, size> vec = *this;

				const int alignedElements = size - size % CMAXDOUBLE;
				const SIMDDouble scalarToVec = BroadcastDouble(static_cast<double>(scalar));

				for (int i = 0; i < alignedElements; i += CMAXDOUBLE) {
					StoreDouble((SIMDDouble*) & (scaled)[i],
						MulDouble(LoadDouble(&vec[i]), scalarToVec)
					);
				}

				auto mask = MaskBySize64<size % CMAXDOUBLE>();
				StoreMaskDouble(&(scaled)[alignedElements], mask,
					MulDouble(
						LoadMaskDouble(&vec[alignedElements], mask),
						scalarToVec
					)
				);
				Vector<T, size> returnValue = scaled;
				return returnValue;
			}

			else {
				Vector<float, size> scaled(0);
				Vector<float, size> vec = *this;

				const int alignedElements = size - size % CMAXFLOAT;
				const SIMDFloat scalarToVec = BroadcastFloat(static_cast<float>(scalar));

				for (int i = 0; i < alignedElements; i += CMAXFLOAT) {
					StoreFloat(&(scaled)[i],
						MulFloat(LoadFloat(&vec[i]), scalarToVec)
					);
				}

				auto mask = MaskBySize32<size % CMAXFLOAT>();
				StoreMaskFloat(&(scaled)[alignedElements], mask,
					MulFloat(
						LoadMaskFloat(&vec[alignedElements], mask),
						scalarToVec
					)
				);
				Vector<T, size> returnValue = scaled;
				return returnValue;
			}			
		}
		
		template <typename U>
		friend Vector<float, size> operator*(const U scalar, const Vector<T, size>& vec);

	private:
	};

	/** Performs scalar multiplication of a vector by a scalar value.
	 *
	 * @details
	 * This function multiplies each element in the vector by a given scalar from the left-hand side and returns a new vector with the results.
	 *
	 * @tparam T The type of the vector.
	 * @tparam size The size of the vector.
	 * @tparam U The type of the scalar value. Must be a floating-point or integer type.
	 * @param scalar The scalar value to multiply.
	 * @param vec The vector to be multiplied.
	 * @return A new Vector<T, size> containing the elements of the original vector multiplied by the scalar.
	 */
	template<typename T, int size, 
		typename U, typename std::enable_if<
		std::is_same<float, U>::value ||
		std::is_same<__int32, U>::value ||
		std::is_same<__int64, U>::value ||
		std::is_same<double, U>::value, int>::type = 0>
	inline Vector<T, size> operator*(const U scalar, Vector<T, size>& vec) {
		return vec.operator*(scalar);
	}

	/**
	 * @brief Performs element-wise multiplication of a scalar value by a vector.
	 *
	 * @details
	 * This function multiplies each element in the vector by a given scalar from the left-hand side and returns a new vector with the results.
	 *
	 * @param scalar The scalar value to multiply.
	 * @param vec The vector to be multiplied.
	 * @return A new vector with each element multiplied by the scalar.
	 */
	template<typename T, int size>
	Vector<T, size> operator*(T scalar, const Vector<T, size>& vec) {
		return vec * scalar; // Uses the previously defined operator*
	}

	/** Merges multiple vectors into a single output vector.
	 * 
	 * @details
	 * This function takes a reference to an output vector and a variadic number of input
	 * vectors. It checks that the output vector's length matches the combined length of
	 * all input vectors before merging their elements into the output vector.
	 *
	 * @tparam T The type of the output vector.
	 * @tparam Vectors The types of the input vectors, which must be compatible.
	 * @param pOutput The output vector where the merged results will be placed.
	 * @param vectors The input vectors to merge into the output vector.
	 * @throws std::out_of_range if the size of the output vector does not match
	 *                             the combined size of the input vectors.
	 */
	template<typename T, typename... Vectors>
	void merge(T& pOutput, Vectors&... vectors) {
		int length = 0;
		([&]
			{
				length += vectors.length();
			} (), ...);
		
		if (pOutput.length() != length) {
			throw std::out_of_range("The size of the output vector does not match the combined size of the input vectors.");
		}

		int mainPosition = 0;
		([&]
			{
				for (const auto element : vectors) {
					pOutput[mainPosition] = element;
					mainPosition++;
				}
			} (), ...);
	}

	/** Type definition for a 4-element integer vector.
	 * 
	 * @details
	 * This typedef creates a 4-element vector of type __int32, allowing for convenient
	 * usage in code. It is useful for representing 4D vector data in various computations.
	 */
	typedef Vector<__int32, 4> IVector4;
	
	/** Type definition for a 4-element float vector.
	 * 
	 * @details
	 * This typedef creates a 4-element vector of type float, enabling easy manipulation
	 * of 4D float vector data. It is commonly used in graphics and mathematical calculations.
	 */
	typedef Vector<float, 4> FVector4;
}
