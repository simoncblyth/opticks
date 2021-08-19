#pragma once

/**

Beware the OptiX limits::

          limitMaxInstanceId :   16777215    ffffff   6*4 = 24 bits  


In [32]: 0x3fff   14bits
Out[32]: 16383

In [19]: "%x" % ( 0x7fff | 0xffc000 )
Out[19]: 'ffffff'

In [21]: 0x3ff
Out[21]: 1023

In [22]: "%x"  % (0x3ff << 10)
Out[22]: 'ffc00'

    1     1
    3    11
    7   111
    f  1111

**/

struct InstanceId
{
    enum { ins_bits = 14, gas_bits = 10 } ; 

    static constexpr unsigned ins_mask = ( 1 << ins_bits ) - 1 ;  
    static constexpr unsigned gas_mask = ( 1 << gas_bits ) - 1 ;  

    static unsigned Encode(unsigned  ins_idx, unsigned  gas_idx ); 
    static void     Decode(unsigned& ins_idx, unsigned& gas_idx, const unsigned identity );     
}; 

inline unsigned InstanceId::Encode(unsigned  ins_idx, unsigned  gas_idx )
{
    assert( ins_idx < ins_mask );
    assert( gas_idx < gas_mask );  
    unsigned identity = (( 1 + ins_idx ) << gas_bits ) | (( 1 + gas_idx ) <<  0 )  ;
    return identity ; 
}

inline void InstanceId::Decode(unsigned& ins_idx, unsigned& gas_idx,  const unsigned identity  )
{
    ins_idx = (( (ins_mask << gas_bits ) & identity ) >> gas_bits ) - 1u ; 
    gas_idx = ((  gas_mask & identity ) >>  0 ) - 1u ;  
}



