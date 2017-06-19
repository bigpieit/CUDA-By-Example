NVCC=nvcc

INC_DIR=/usr/local/cuda-8.0/samples/common/inc/

NVCC_OPTs=-arch=sm_20 -I$(INC_DIR)

DevicePointer:DevicePointer.o
	$(NVCC) -o DevicePointer DevicePointer.o

DevicePointer.o:DevicePointer.cu $(INC_DIR)helper_cuda.h $(INC_DIR)helper_functions.h
	$(NVCC) $(NVCC_OPTs) -c DevicePointer.cu

clean:
	rm DevicePointer DevicePointer.o
