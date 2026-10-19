#pragma once
/**
sFFS.h
=======

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#  if defined(__CUDA_ARCH__)
#    define FFS(x)   (__ffs(x))
#    define FFSLL(x) (__ffsll(x))
#  else
#    // Host compilation path under nvcc
#    define FFS(x)   (__builtin_ffs(x))
#    define FFSLL(x) (__builtin_ffsll(x))
#  endif
#else
// Pure host compilation (g++/clang++)
#define FFS(x)   (__builtin_ffs(x))
#define FFSLL(x) (__builtin_ffsll(x))
#endif

#define rFFS(x)   ( x == 0 ? 0 : 0x1 << (x - 1) )
#define rFFSLL(x) ( x == 0 ? 0 : 0x1ull << (x - 1) )
// use rFFSLL when values of x exceed about 31


