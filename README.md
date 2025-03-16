# Syren Math
A hobby vector and matrix c++ math library (development in-progress) that incorporates SIMD instructions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Prerequisites
- C++20 or higher
- Compiled on MSVC 
- Windows operating system
- CPU with AVX2 support

## Installation
1. Clone the repository: https://github.com/AshvinPerera/Syren-Math.git
2. Add "Syren Math" folder to include path
3. Define the appropriate preprocessor macro depending on SIMD support (e.g: USING_AVX2)

## Usage
To use the math library, include the relevent header file of the feature you would like to use. 
```c++
#include "Vector.h"
#include "Matrix.h"
```

Note: all library definitions exist inside the SyrenMath namespace

## Example
```c++
    SyrenMath::Vector<__int32, 102> x(1);
    SyrenMath::Vector<__int32, 102> y(2);
    
    SyrenMath::Vector<__int32, 102> z = Vec1 + Vec2;
    SyrenMath::Vector<__int64, 102> w = Vec1;

    __int64 dotProduct = w.t() * w
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
