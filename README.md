# Fast 3x3 SVD

This is an implementation of the method described in
[Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations](http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf),
based on an implementation by Eric Jang. I've updated the syntax of the library to
reflect a more modern c++ style, and included Python bindings an unit tests.

## Usage

Just include the header file and you are good to go!

### C++

```cpp
#include "svd3.h"
double a[3][3] = {
  {-0.558253, -0.0461681, -0.505735},
  {-0.411397, 0.0365854, 0.199707},
  {0.285389, -0.313789, 0.200189}
};

double u[3][3], s[3][3], v[3][3];

svd(a, u, s, v);
```

### Python

Import the `svd` method from the library.

```py
from svd3x3 import svd

a = [
    [-0.558253, -0.0461681, -0.505735],
    [-0.411397, 0.0365854, 0.199707],
    [0.285389, -0.313789, 0.200189],
]
svd(a)
```

## Performance

### Running the Benchmarks

Benchmarks for the C and Python APIs can be run as follows:

```sh
# Python
python -c "import svd3x3; svd3x3._bench()"

# C++
clang++ -O3 bench.cpp -o bench # To benchmark the original library, set -DBENCH_ORIGINAL
ulimit -s 65500 # Required to prevent stack overflow, otherwise decrease N_SAMPLES
./bench
```

### Benchmark Results

All execution time tests were evaluated on an M1 Pro, with CPU temperature maintained
below 55°C for the entire duration.

|       Python API       |     Numpy SVD      | % Improvement |
| :--------------------: | :----------------: | :-----------: |
| **2.8460 ± 0.0304 μs** | 7.6621 ± 0.0158 μs |     62.9%     |

| C API (`double`, `-03`) | Original SVD3 Library (`float`, `-03`) | % Improvement |
| :---------------------: | :------------------------------------: | :-----------: |
| **0.4725 ± 0.0025 μs**  |          0.540434 ± 0.0021 μs          |     12.6%     |

Our implementation is both faster and more accurate than the original C source despite
using double precision, and is significantly faster than Numpy.

## License

MIT License, Eric V. Jang 2014, Jenna Bradley 2025
