#pragma once

#define USING_AVX2
#ifdef USING_AVX2
#include "AVX2.h"
#endif

#include "Vector.h"


namespace SyrenMath {
	template<typename T, unsigned int m, unsigned int n>
	class Matrix {
		std::vector<Vector> mData;
	};
}
