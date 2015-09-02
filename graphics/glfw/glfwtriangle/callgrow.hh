#pragma once

template <typename T> class CudaGLBuffer ; 

void callgrow(CudaGLBuffer<float3>* cgb, unsigned int n);

