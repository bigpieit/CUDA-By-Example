// includes, system
#include <stdlib.h>
#include <stdio.h>


// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>

#include "../common/cpu_bitmap.h"
#include "../common/book.h"

#define DIM 1000

struct cuComplex {
  float r;
  float i;
  __device__ cuComplex( float a, float b) : r(a), i(b) {}

  __device__ float magnitude2( void ) {
    return r * r + i * i;
  }
  __device__ cuComplex operator*(const cuComplex& a) {
    return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
  __device__ cuComplex operator+(const cuComplex& a) {
    return cuComplex(r+a.r, i+a.i);
  }
};

__device__ int julia( int x, int y) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
  float jy = scale * (float)(DIM/2 - y)/(DIM/2);

  cuComplex c(-0.8,0.154);
  cuComplex a(jx,jy);

  int i = 0;
  for (i=0; i<200; i++) {
    a = a*a + c;
    if (a.magnitude2() > 1000)
      return 0; // return 0 if it is not in set
  }
  return 1; // return 1 if point is in set
}


__global__ void kernel( unsigned char *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x+y*gridDim.x;

  // now calculate the value at that position
  int juliaValue = julia(x,y);
  ptr[offset*4 + 0] = 255 * juliaValue;  // red if julia() returns 1, black if pt. not in set
  ptr[offset*4 + 1] = 0;
  ptr[offset*4 + 2] = 0;
  ptr[offset*4 + 3] = 255;
}

int main(void) {
  CPUBitmap bitmap( DIM, DIM );
  unsigned char *dev_bitmap;

  checkCudaErrors(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

  dim3 grid(DIM,DIM);

  kernel<<<grid,1>>>(dev_bitmap);

  checkCudaErrors(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost ));
  
  bitmap.display_and_exit();
  printf("size %ld is done\n", bitmap.image_size());
  checkCudaErrors( cudaFree(dev_bitmap));
}