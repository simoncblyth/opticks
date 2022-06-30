// ./logTest.sh

#include "NP.hh"

__global__ void test_log_(double* dd)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nx = blockDim.x ; 

    double d = double(ix)/double(nx-1) ; 
    float  f = float(d) ;

    double d0 = -1.*log( d );
    float  f0 = -1.f*log( f );

    dd[ix*4+0] = d ; 
    dd[ix*4+1] = d0 ; 
    dd[ix*4+2] = f0 ; 
    dd[ix*4+3] = 0. ; 

    //printf("//test_log  (ix,iy,nx) (%2d, %2d, %2d) \n", ix, iy, nx );
}

void test_log()
{
    unsigned ni = 1001 ; 
    unsigned nj = 4 ; 

    dim3 block(ni,1); 
    dim3 grid(1,1);

    NP* h = NP::Make<double>( ni, nj ) ; 
    unsigned arr_bytes = h->arr_bytes() ; 
    double* hh = h->values<double>(); 

    double* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    test_log_<<<grid,block>>>(dd);  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save("/tmp/logTest.npy"); 
}

int main()
{
    test_log();
    return 0 ; 
}


