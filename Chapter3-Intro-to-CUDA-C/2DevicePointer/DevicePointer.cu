// includes, system
#include <stdlib.h>
#include <stdio.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void add(int a, int b, int *c) {
	*c = a + b;
}

int main (void) {
	int c;
	int *dev_c;
	
	checkCudaErrors(cudaMalloc((void **) &dev_c, sizeof(int)));

	add<<<1,1>>>(2, 7, dev_c);
	
	checkCudaErrors(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2 + 7 = %d\n", c);

	cudaFree(dev_c);

	return 0;
}