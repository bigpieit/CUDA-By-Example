NVCC=nvcc

INC_DIR=/usr/local/cuda-8.0/samples/common/inc/

NVCC_OPTs=-arch=sm_20 -I$(INC_DIR) -Wno-deprecated-gpu-targets

CudaProp:CudaProp.o
	$(NVCC) -Wno-deprecated-gpu-targets -o CudaProp CudaProp.o

CudaProp.o:CudaProp.cu $(INC_DIR)helper_cuda.h $(INC_DIR)helper_functions.h
	$(NVCC) $(NVCC_OPTs) -c CudaProp.cu

clean:
	rm CudaProp CudaProp.o