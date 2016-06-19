#include "BBit.hh"

#include <cstring>

#if defined(_MSC_VER)

#include <intrin.h>

int BBit::ffs(int i)
{
    // https://msdn.microsoft.com/en-us/library/wfd9z0bb.aspx
    unsigned long mask = i ;
    unsigned long index ;
    unsigned char masknonzero = _BitScanForward( &index, mask );
    return masknonzero ? index + 1 : 0 ;
}

#elif defined(__MINGW32__)

int BBit::ffs(int i)
{
   return __builtin_ffs(i);
}

#else

int BBit::ffs(int i)
{
   return ::ffs(i);
}

#endif


