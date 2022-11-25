#pragma once

/**
https://stackoverflow.com/questions/1392059/algorithm-to-generate-bit-mask
**/

#include <climits>

template <typename T>
static constexpr T sbitmask(unsigned n)
{
    //return static_cast<T>(-(n != 0)) & (static_cast<T>(-1) >> ((sizeof(T) * CHAR_BIT) - n));
    return n == 0 ? 0 : (static_cast<T>(-1) >> ((sizeof(T) * CHAR_BIT) - n));
}

/**
sbitmask_ = lambda i:np.uint64(-1) >> np.uint64(64-i) 

In [51]: for i in range(64+1):print(" {0:2d} {1:064b} ".format(i, sbitmask_(i)))                                                                                            
  0 0000000000000000000000000000000000000000000000000000000000000000 
  1 0000000000000000000000000000000000000000000000000000000000000001 
  2 0000000000000000000000000000000000000000000000000000000000000011 
  3 0000000000000000000000000000000000000000000000000000000000000111 
  4 0000000000000000000000000000000000000000000000000000000000001111 
  5 0000000000000000000000000000000000000000000000000000000000011111 
  6 0000000000000000000000000000000000000000000000000000000000111111 

In [61]: "{0:064b}".format( ~sbitmask_(64-8) )
Out[61]: '1111111100000000000000000000000000000000000000000000000000000000'

In [62]: "{0:064b}".format( ~sbitmask_(64-7) )
Out[62]: '1111111000000000000000000000000000000000000000000000000000000000'

In [63]: "{0:064b}".format( ~sbitmask_(64-3) )
Out[63]: '1110000000000000000000000000000000000000000000000000000000000000'


**/


template <typename T>
static constexpr T sbitmask_0(unsigned n)
{
    return static_cast<T>(-(n != 0)) ;   
    // all bits set, except for n=0 which gives all bits unset  
}

template <typename T>
static constexpr T sbitmask_1(unsigned n)
{
   return static_cast<T>(-1) >> ((sizeof(T) * CHAR_BIT) - n);
}

template <typename T>
static constexpr T sbitmask_2(unsigned n)
{
   return static_cast<T>(-1) ;
}


