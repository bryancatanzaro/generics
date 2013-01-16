LDG
===

NVIDIA GPUs of CUDA compute capability 3.5 and greater support
`__ldg()`, an intrinsic that loads through the read-only texture
cache, and can improve performance in some circumstances.  CUDA
provides overloads of `__ldg()` for some built-in types.

This library provides a single template:

    template<typename T> __device__ T __ldg(const T*);

That allows data of any type to be loaded using `__ldg`. The only
restriction on `T` is that it have a default constructor.

Usage
=====

To use this library, simply `#include "ldg.h"`.  
The `__ldg()` overloads provided natively by CUDA will be used if `T`
is natively supported.  If not, the template will be used.

See
[test.cu](http://github.com/BryanCatanzaro/ldg/blob/master/test.cu)
for an example.