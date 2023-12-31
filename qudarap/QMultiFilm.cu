

#include "curand_kernel.h"
#include "scuda.h"
#include "squad.h"
#include "qmultifilm.h"
//#include "qscint.h"
#include "stdio.h"

/**

**/

__global__ void _QMultiFilm_check(unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
   // printf("hello world!");
    if( ix % 10 == 0 )   printf( "//_QMultiFilm_check  ix %d iy %d width %d height %d \n", ix, iy, width, height ); 
}

__global__ void _QMultiFilm_lookup(cudaTextureObject_t tex, quad4* meta, float4* lookup, unsigned num_lookup, unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int index = iy * width + ix;
    unsigned nx = meta->q0.u.x ; 
    unsigned ny = meta->q0.u.y ; 

    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;

    float4 v = tex2D<float4>( tex, x, y );     

    //printf( "//_QMultiFilm_lookup  ix %d iy %d width %d height %d index %d nx %d ny %d x %10.4f y %10.4f v.x %10.4f v.y %10.4f v.z %10.4f v.w %10.4f  \n", ix, iy, width, height, index, nx, ny, x, y, v.x, v.y, v.z, v.w ); 
    lookup[ index ] = v ;  
}

__global__ void _QMultiFilm_mock_lookup(qmultifilm* d_multifilm, quad2* d_input, float4* d_out, unsigned num_lookup, unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int index = iy * width + ix;
	int pmtcat  = d_input[index].q0.i.x;
	float wv_nm = d_input[index].q0.f.y;
	float aoi   = d_input[index].q0.f.z;

	float Rs = d_input[index].q1.f.x;
	float Ts = d_input[index].q1.f.y;
	float Rp = d_input[index].q1.f.z;
	float Tp = d_input[index].q1.f.w;

	float4 value = d_multifilm->lookup(pmtcat, wv_nm, aoi);
    d_out[ index ] = value ;  
	
	int item = 10;
	if(index < item || index > (num_lookup - item)){
    	printf( "//_QMultiFilm_mock_lookup  ix %d iy %d width %d height %d index %d Rs %10.4f Ts %10.4f Rp %10.4f Tp %10.4f Rs_i %10.4f Ts_i %10.4f Rp_i %10.4f Tp_i %10.4f \n", ix, iy, width, height, index, Rs, Ts, Rp, Tp, value.x, value.y , value.z, value.w ); 
	}
}
extern "C" void QMultiFilm_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t tex, quad4* meta, float4* lookup, unsigned num_lookup, unsigned width, unsigned height) 
{
    printf("//QMultiFilm_lookup num_lookup %d width %d height %d \n", num_lookup, width, height);  
    _QMultiFilm_lookup<<<numBlocks,threadsPerBlock>>>( tex, meta, lookup, num_lookup, width, height );
} 


extern "C" void QMultiFilm_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height ) 
{
    printf("//QMultiFilm_check width %d height %d threadsPerBlock.x %d threadsPerBlock.y %d  numBlocks.x %d numBlocks.y %d \n", width, height,threadsPerBlock.x, threadsPerBlock.y , numBlocks.x, numBlocks.y);  

    _QMultiFilm_check<<<numBlocks,threadsPerBlock>>>( width, height  );
} 


extern "C" void QMultiFilm_mock_lookup(dim3 numBlocks, dim3 threadsPerBlock, qmultifilm* d_multifilm, quad2* d_input, float4* d_out, unsigned num_lookup, unsigned width, unsigned height) 
{
    printf("//QMultiFilm_mock_lookup num_lookup %d width %d height %d \n", num_lookup, width, height);  
    _QMultiFilm_mock_lookup<<<numBlocks,threadsPerBlock>>>( d_multifilm, d_input, d_out, num_lookup, width, height );
} 


