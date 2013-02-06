CUDA_ARCH ?= sm_35

test: test.cu
	nvcc -arch=$(CUDA_ARCH) -o test test.cu -I..