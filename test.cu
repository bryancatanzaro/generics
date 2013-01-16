#include "ldg.h"

#include <thrust/device_vector.h>
#include <iostream>

template<typename T>
struct non_pod {
    T x;
    T y;
    T z;
    __host__ __device__
    non_pod() {}
    __host__ __device__
    non_pod(T a, T b, T c) : x(a), y(b), z(c) {} 
    friend std::ostream& operator<<(std::ostream& os, const non_pod& f) {
        os << (int)f.x << " " << (int)f.y << " " << (int)f.z;
        return os;
    }
};



template<typename T>
__global__ void test_ldg(const T* i, T* o) {
    *o = __ldg(i);
}

int main() {
    typedef non_pod<char> non_pod3;
    thrust::device_vector<non_pod3> i3(1);
    thrust::device_vector<non_pod3> o3(1);
    i3[0] = non_pod3(1,2,3);
    test_ldg<<<1,1>>>(thrust::raw_pointer_cast(i3.data()),
                      thrust::raw_pointer_cast(o3.data()));
    non_pod3 r3 = o3[0];
    std::cout << r3 << std::endl;
    typedef non_pod<int> non_pod12;
    thrust::device_vector<non_pod12> i12(1);
    thrust::device_vector<non_pod12> o12(1);
    i12[0] = non_pod12(4,5,6);
    test_ldg<<<1,1>>>(thrust::raw_pointer_cast(i12.data()),
                      thrust::raw_pointer_cast(o12.data()));
    non_pod12 r12 = o12[0];
    std::cout << r12 << std::endl;    
}
    

