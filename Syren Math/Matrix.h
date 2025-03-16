#pragma once

#ifdef USING_AVX2
#include "AVX2.h"
#endif

#include "Vector.h"


namespace SyrenMath {
	/** Represents a mathematical matrix of dimensions `m x n`.
	 *
	 * @details
	 * This template-based matrix class utilizes `Vector<T, m>` as column representations,
	 * storing data efficiently in a column-major format. The matrix can be used for mathematical
	 * operations such as multiplication, transformations, and other numerical computations.
	 *
	 * @tparam T The data type of the matrix elements (e.g., `float`, `double`, `int`).
	 * @tparam m The number of rows in the matrix.
	 * @tparam n The number of columns in the matrix.
	 */
	template<typename T, unsigned int m, unsigned int n>
	class Matrix {
		std::vector<Vector<T, m>> mData;
	};
}
