// ./SIMGStandaloneTest.sh

#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#include <iostream>
#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"

// https://stackoverflow.com/questions/14901491/cudamemcpytoarray/14929827#14929827

#include <stdio.h>
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)



__global__ void colorKernel(uchar4* output, cudaTextureObject_t texObj, int width, int height, float theta) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //if( x % 1000 == 0 ) printf("x %d y %d \n", x, y ); 
    //if( x == 1000 ) printf("x %d y %d \n", x, y ); 

	output[y * width + x] = make_uchar4( 255u, 0u, 0u, 255u ); 
}



__global__ void transformKernel(uchar4* output, cudaTextureObject_t texObj, int width, int height, float theta) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
    float u = x / (float) width;  // 0. -> 1. 
	float v = y / (float) height;

    // shift origin to center of image
	u -= 0.5f;                   //  -0.5 -> 0.5 
	v -= 0.5f;

    // rotate around the center
	float tu = u * cosf(theta) - v * sinf(theta) ;
	float tv = v * cosf(theta) + u * sinf(theta) ;

    // read from the texture  
    uchar4 c = tex2D<uchar4>(texObj, tu+0.5f, tv+0.5f); 

    //if( c.x != 0 ) printf(" c ( %d %d %d %d ) \n",c.x, c.y, c.z, c.w );  
    //c.x = 255u ; 
    c.w = 255u ; 

	output[y * width + x] = c ;
}

int main(int argc, char** argv)
{
    const char* ipath = argc > 1 ? argv[1] : "/tmp/i.png" ; 
    const char* opath = argc > 2 ? argv[2] : "/tmp/o.png" ; 

    SIMG img(ipath); 
    std::cout << img.desc() << std::endl ; 
    assert( img.channels == 4 ); 

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, img.width, img.height );
    cudaCheckErrors("cudaMallocArray");

    cudaMemcpyToArray(cuArray, 0, 0, img.data, img.width*img.height*4*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpyToArray");

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaTextureDesc.html
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;

	//texDesc.filterMode = cudaFilterModeLinear;
	texDesc.filterMode = cudaFilterModePoint;    // switch off interpolation, as that gives error with non-float texture  

	texDesc.readMode = cudaReadModeElementType;  // return data of the type of the underlying buffer
	texDesc.normalizedCoords = 1 ;            // addressing into the texture with floats in range 0:1

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	// Allocate result of transformation in device memory
	uchar4* d_output;
	cudaMalloc(&d_output, img.width * img.height * 4*sizeof(unsigned char));

	dim3 dimBlock(16, 16);
	dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

    float theta = 1.f ; 

	//colorKernel<<<dimGrid, dimBlock>>>(d_output, texObj, img.width, img.height, theta );
	transformKernel<<<dimGrid, dimBlock>>>(d_output, texObj, img.width, img.height, theta );
    cudaDeviceSynchronize();      
    cudaCheckErrors("cudaDeviceSynchronize"); 
    // Fatal error: cudaDeviceSynchronize (linear filtering not supported for non-float type at SIMGStandaloneTest.cu:123)


    uchar4* output = new uchar4[img.width*img.height] ; 
    cudaMemcpy(output, d_output, img.width*img.height*sizeof(uchar4), cudaMemcpyDeviceToHost);     

    std::cout << "writing to " << opath << std::endl ; 

    SIMG img2(img.width, img.height, img.channels, (unsigned char*)output ); 
    img2.writePNG(opath); 

    cudaDeviceSynchronize();  

	cudaDestroyTextureObject(texObj);
	cudaFreeArray(cuArray);

    delete[] output ; 
	cudaFree(d_output);

    return 0;
}

