Generics
===

This library generalizes certain CUDA intrinsics to work on arbitrary
data types.  For example, NVIDIA GPUs of CUDA compute capability 3.5
and greater, such as the [Tesla
K20](http://www.nvidia.com/object/personal-supercomputing.html),
support `__ldg()`, an intrinsic that loads through the read-only
texture cache, and can improve performance in some circumstances.
This library allows `__ldg` to work on arbitrary types, as detailed
below. It also generalizes `__shfl()` to shuffle arbitrary types.

LDG
===

CUDA provides overloads of `__ldg()` for some built-in types:

`char`, `short`, `int`, `long long`, `int2`, `int4`, `unsigned
char`, `unsigned short`, `unsigned int`, `unsigned long long`,
`uint2`, `uint4`, `float`, `double`, `float2`, `float4`, `double2`.

However, for all other types, including user defined types, the native
overloads of `__ldg()` are insufficient.  To solve this problem, this
library provides a template:

    template<typename T> __device__ T __ldg(const T*);

This template allows data of any type to be loaded using `__ldg`. The
only restriction on `T` is that it have a default constructor.


To use this library, simply `#include <generics/ldg.h>`.  
The `__ldg()` overloads provided natively by CUDA will be used if `T`
is natively supported.  If not, the template will be used.

See
[ldg.cu](http://github.com/BryanCatanzaro/generics/blob/master/test/ldg.cu)
for an example.

If you are compiling for CUDA compute capability of less than 3.5,
`__ldg()` will fall back to traditional loads.

SHFL
====

For devices of compute capability 3.0 or above, CUDA provides a
set of `__shfl()` intrinsics that share data between threads in a warp,
without using any shared memory.  CUDA provides overloads for `int`
and `float` types.  For all other types, this library provides a few
templates:

    template<typename T> __device__ T __shfl(const T& t, const int& i);
    template<typename T> __device__ T __shfl_down(const T& t, const int& delta);
    template<typename T> __device__ T __shfl_up(const T& t, const int& delta);
    template<typename T> __device__ T __shfl_xor(const T& t, const int& mask);

This allows data of other types to be shuffled using the `__shfl()`
mechanism. There are two restrictions on `T`:

 * `sizeof(T)` must be divisible by 4. The code will fail to compile
      if you instantiate it with a type that does not satisfy this
      requirement.
 * `T` must have a default constructor

To use this library, simply `#include <generics/shfl.h>`.  
The `__shfl()` overloads provided natively by CUDA will be used if `T`
is natively supported.  If not, the template will be used.

 
See
[shfl.cu](http://github.com/BryanCatanzaro/generics/blob/master/test/shfl.cu)
for an example.