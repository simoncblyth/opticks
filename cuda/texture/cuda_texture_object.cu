/*
    http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

    Kepler GPUs and CUDA 5.0 introduce a new feature called texture objects
    (sometimes called bindless textures, since they don’t require manual
    binding/unbinding) that greatly improves the usability and programmability of
    textures. Texture objects use the new cudaTextureObject_t class API, whereby
    textures become first-class C++ objects and can be passed as arguments just as
    if they were pointers.  There is no need to know at compile time which textures
    will be used at run time, which enables much more dynamic execution and
    flexible programming, as shown in the following code.

    Need to compile with at least compute capability 3.0 ie with:   -arch=sm_30 

*/

#define N 1024

// texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float x = tex1Dfetch<float>(tex, i);
  // do some work using x ...
}

void call_kernel(cudaTextureObject_t tex) {
  dim3 block(128,1,1);
  dim3 grid(N/block.x,1,1);
  kernel <<<grid, block>>>(tex);
}

int main() {
  // declare and allocate memory
  float *buffer;
  cudaMalloc(&buffer, N*sizeof(float));

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = buffer;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32; // bits per channel
  resDesc.res.linear.sizeInBytes = N*sizeof(float);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  cudaTextureObject_t tex=0;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  call_kernel(tex); // pass texture as argument

  // destroy texture object
  cudaDestroyTextureObject(tex);

  cudaFree(buffer);
}
