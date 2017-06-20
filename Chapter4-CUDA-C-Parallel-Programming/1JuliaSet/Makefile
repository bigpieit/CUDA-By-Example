NVCC=nvcc

INC_DIR1=/usr/local/cuda-8.0/samples/common/inc/
INC_DIR2=../../common/

NVCC_OPTs=-arch=sm_20 -I$(INC_DIR1) -I$(INC_DIR2) -lglut -lGLU -lGL -Wno-deprecated-gpu-targets

Julia:Julia.o
	$(NVCC) $(NVCC_OPTs) -Wno-deprecated-gpu-targets -o Julia Julia.o

Julia.o:Julia.cu $(INC_DIR1)helper_cuda.h $(INC_DIR1)helper_functions.h $(INC_DIR2)cpu_bitmap.h $(INC_DIR2)book.h
	$(NVCC) $(NVCC_OPTs) $(LIBS) -c Julia.cu

clean:
	rm Julia Julia.o