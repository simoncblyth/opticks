#pragma once

#include <strings.h>

struct sbit
{
    static int ffs(int i);
#if defined(_MSC_VER)
#elif defined(__MINGW32__)
#else
    static long long ffsll(long long i);
#endif
};


#if defined(_MSC_VER)

#include <intrin.h>

inline int sbit::ffs(int i)
{
    // https://msdn.microsoft.com/en-us/library/wfd9z0bb.aspx
    unsigned long mask = i ;
    unsigned long index ;
    unsigned char masknonzero = _BitScanForward( &index, mask );
    return masknonzero ? index + 1 : 0 ;
}

#elif defined(__MINGW32__)

inline int sbit::ffs(int i)
{
   return __builtin_ffs(i);
}

#else

inline int sbit::ffs(int i)
{
   return ::ffs(i);
}

inline long long sbit::ffsll(long long i)
{
   return ::ffsll(i);
}


#endif


