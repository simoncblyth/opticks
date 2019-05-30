#pragma once

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"


class BRAP_API BBit {
    public:
         // ffs returns 1-based index of rightmost set bit, see man ffs 
        static int ffs(int msk);
        static long long ffsll(long long msk);

        static unsigned long long count_nibbles(unsigned long long x); 



}; 

#include "BRAP_TAIL.hh"


