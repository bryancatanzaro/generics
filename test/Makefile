CUDA_ARCH ?= sm_35

ldg: ldg.cu
	nvcc -arch=$(CUDA_ARCH) -o ldg ldg.cu -I..