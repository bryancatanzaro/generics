#include <thrust/device_vector.h>
#include <iostream>


#include <generics/shfl.h>

//Just for example purposes - a non plain old data type
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


//Uses SHFL to move elements
template<typename T>
__global__ void test_shfl(T* o) {
    int idx = threadIdx.x;
    T data(idx, idx+1, idx+2);
    o[idx] = __shfl(data, 15);
}

int main() {   
    //sizeof(non_pod<int>) is 12
    //Therefore we can use the generic __shful
    typedef non_pod<int> non_pod12;
    thrust::device_vector<non_pod12> o12(32);
    test_shfl<<<1,32>>>(thrust::raw_pointer_cast(o12.data()));
    non_pod12 r12 = o12[0];
    std::cout << r12 << std::endl;    
}
    

