// dir=/tmp/$USER/opticks/cudarap && name=cuda_pointer_arithmetic && mkdir -p $dir && nvcc -o $dir/$name -std=c++11 $name.cu &&  $dir/$name
/**
https://stackoverflow.com/questions/5909485/cuda-device-pointer-manipulation

**/

#include <vector>
#include <cstdio>


void test_offset_device_pointer_0()
{
    const int na = 5, nb = 4;
    float a[na] = { 1.2, 3.4, 5.6, 7.8, 9.0 };
    float b[nb] = { 0.0, 0.0, 0.0, 0.0 } ; 

    size_t sz_a = size_t(na) * sizeof(float);
    size_t sz_b = size_t(nb) * sizeof(float);

    float* d_a = NULL ; 
    cudaMalloc((void **)&d_a, sz_a );
    float* d_a_plus_1 = d_a + 1 ; 

    cudaMemcpy( d_a,          a,  sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy( b  , d_a_plus_1,  sz_b, cudaMemcpyDeviceToHost);

    for(int i=0; i<nb; i++) printf("%d %f\n", i, b[i]);

}

void test_device_array_offsets()
{
    std::vector<float> bbs = { 
          -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, 
          -2.0f, -2.0f, -2.0f, +2.0f, +2.0f, +2.0f, 
          -3.0f, -3.0f, -3.0f, +3.0f, +3.0f, +3.0f, 
          -4.0f, -4.0f, -4.0f, +4.0f, +4.0f, +4.0f, 
          -5.0f, -5.0f, -5.0f, +5.0f, +5.0f, +5.0f, 
          -6.0f, -6.0f, -6.0f, +6.0f, +6.0f, +6.0f,
          -7.0f, -7.0f, -7.0f, +7.0f, +7.0f, +7.0f,
          -8.0f, -8.0f, -8.0f, +8.0f, +8.0f, +8.0f,
          -9.0f, -9.0f, -9.0f, +9.0f, +9.0f, +9.0f
         };

    size_t num_float = size_t(bbs.size()) ; 
    size_t num_bytes = num_float*sizeof(float) ;

    float* d_bb = NULL ; 
    cudaMalloc((void **)&d_bb, num_bytes );
    cudaMemcpy( d_bb,  bbs.data(),  num_bytes, cudaMemcpyHostToDevice);

    float bbi[6] ; 
    for(unsigned i=0 ; i < 9 ; i++)
    {
        cudaMemcpy( bbi  , d_bb + 6*i , 6*sizeof(float), cudaMemcpyDeviceToHost);
        printf("%d :", i );  
        for(int j=0; j<6 ; j++) printf("%f ", bbi[j]);
        printf("\n");  
    }
}


int main(void)
{
    cudaFree(0);

    //test_offset_device_pointer_0(); 
    test_device_array_offsets();

    cudaThreadExit();
}
