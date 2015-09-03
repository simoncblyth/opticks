#pragma once

template <typename T> class CudaGLBuffer ; 

void callgrow_index(CudaGLBuffer<float3>* cgb, unsigned int n, bool mapunmap);
void callgrow_value(CudaGLBuffer<float3>* cgb, unsigned int n, bool mapunmap);

