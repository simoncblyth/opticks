

__global__ void QRng_transformKernel_(uchar4* output, cudaTextureObject_t texObj, size_t width, size_t height, float theta) 
{
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


extern "C" void transformKernel(dim3 dimGrid, dim3 dimBlock, uchar4* d_output, cudaTextureObject_t texObj,  size_t width, size_t height, float theta )
{
    QRng_transformKernel_<<<dimGrid,dimBlock>>>(d_output, texObj, width, height, theta);
}



