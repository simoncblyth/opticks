// nvcc -arch=sm_20 -o t557 t557.cu && ./t557 && rm ./t557

/*

http://stackoverflow.com/questions/25827825/offset-parameter-of-curand-init

curand_init(seed, idx+2, 0, &state);

should not be equivalent to:

curand_init(seed, idx, 2, &state);

because sequences generated with the same seed and different sequence numbers
will not have statistically correlated values. Note that there is no additional
qualification of this statement based on the offset parameter.

The overall sequence generated has a period greater than 2^190. Within this
overall sequence there are subsequences that are identified by the 2nd
parameter of the curand_init call. These subsequences are independent from each
other (and not statistically correlated). These subsequences are approximately
2^67 numbers apart in the overall sequence. The offset parameter (3rd
parameter) selects the starting position within this subsequence. Since the
offset parameter cannot be larger than 2^67, it's not possible to use the
offset parameter by itself to cause generated numbers to overlap.

However there is a skipahead_sequence function provided which can allow us to
do this, if we choose. Consider the following modification to your code and
sample run:

*/


#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCKSIZE 256

/**********/
/* iDivUp */
/**********/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/********************************************************/
/* KERNEL FUNCTION FOR TESTING RANDOM NUMBER GENERATION */
/********************************************************/
__global__ void testrand1(unsigned long seed, float *a, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    if (idx < N) {
        curand_init(seed, idx, 0, &state);
        a[(idx*2)] = curand_uniform(&state);
        if(idx%2)
          skipahead_sequence(1, &state);
        a[(idx*2)+1] = curand_uniform(&state);

    }
}

/********/
/* MAIN */
/********/
int main() {

    const int N = 10;

    float *h_a  = (float*)malloc(2*N*sizeof(float));
    float *d_a; gpuErrchk(cudaMalloc((void**)&d_a, 2*N*sizeof(float)));

    testrand1<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(1234, d_a, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_a, d_a, 2*N*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<2*N; i++) printf("%i %f\n", i, h_a[i]);

}

