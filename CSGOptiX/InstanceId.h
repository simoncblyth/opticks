#pragma once
/**
InstanceId_NOT_IN_USE
======================

Initially planned to bitpack inside InstanceId but turns out to not be very useful::

    unsigned instance_id = optixGetInstanceId() ;  // see IAS_Builder::Build and InstanceId.h 

Due to the OptiX 7 limits and the flat nature of ins_idx in single IAS approach::

    limitMaxInstanceId :   16777215    ffffff   6*4 = 24 bits    ~16.77M  

This was developed whilst unaware of the 24bit optix limitation : limitMaxInstanceId 

JUNO geometry, has the flat instance id reaching to : 46116   
(this may be with struts skipped, so full total could be a few thousands more than that)
see SBT::dumpIAS

So could use 16 bits for that, leaving 8 bits going spare in the. But thats
too tight (255) for general gas_idx. So probably not much utility in 
bitpacking in the instanceId:: 


    In [32]: 0x3fff   14bits
    Out[32]: 16383

    In [12]: 0x7fff   15bits 
    Out[12]: 32767

    In [11]: 0xffff   16 bits    
    Out[11]: 65535


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


The above is thinking in terms of the old model with split pots 
of transforms for each instance. But with optix7 are currently 
using a single IAS with all transforms of all instances in it.   
So 14 bits is not enough, would need to use 16 bits for JUNO flat instances. 

8-bits (255) could hold gas_idx : but its a bit too tight to be a general approach, 
and there is actually no need to do that as can use the flat instance_idx to lookup 
the instrumented transform giving gas_idx that way. 

**/

struct InstanceId_NOT_IN_USE
{
    //enum { ins_bits = 14, gas_bits = 10 } ; 
    enum { ins_bits = 16, gas_bits = 8 } ; 

    static constexpr unsigned ins_mask = ( 1 << ins_bits ) - 1 ;  
    static constexpr unsigned gas_mask = ( 1 << gas_bits ) - 1 ;  

    static unsigned Encode(unsigned  ins_idx, unsigned  gas_idx ); 
    static void     Decode(unsigned& ins_idx, unsigned& gas_idx, const unsigned identity );     
}; 

inline unsigned InstanceId_NOT_IN_USE::Encode(unsigned  ins_idx, unsigned  gas_idx )
{
    assert( ins_idx < ins_mask );
    assert( gas_idx < gas_mask );  
    unsigned identity = (( 1 + ins_idx ) << gas_bits ) | (( 1 + gas_idx ) <<  0 )  ;
    return identity ; 
}

inline void InstanceId_NOT_IN_USE::Decode(unsigned& ins_idx, unsigned& gas_idx,  const unsigned identity  )
{
    ins_idx = (( (ins_mask << gas_bits ) & identity ) >> gas_bits ) - 1u ; 
    gas_idx = ((  gas_mask & identity ) >>  0 ) - 1u ;  
}



