CUDA_ARCH ?= sm_35

ALL: ldg shfl

ldg: ldg.cu
	nvcc -arch=$(CUDA_ARCH) -o ldg ldg.cu -I..

shfl: shfl.cu
	nvcc -arch=$(CUDA_ARCH) -o shfl shfl.cu -I..