test: test.cu ldg.h array.h fuse.h
	nvcc -arch=sm_35 -o test test.cu