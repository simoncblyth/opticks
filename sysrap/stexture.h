#pragma once

#include "scuda.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
   #include "s_mock_texture.h"
#endif

#endif 


