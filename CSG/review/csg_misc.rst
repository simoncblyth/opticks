review/csg_misc
================




csg_pack.h : unsigned PACK4/UNPACK4 - used for storing postorder indices sequences
-------------------------------------------------------------------------------------

::

    #define PACK4( a, b, c, d)   ( \
           (( (a) & 0xff ) <<  0 ) | \
           (( (b) & 0xff ) <<  8 ) | \
           (( (c) & 0xff ) << 16 ) | \
           (( (d) & 0xff ) << 24 ) \
                                 )


* better and simpler to use uchar4 or uchar2 or ushort2 - for storing slices


csg_tranche.h : tr.curr starts at -1, becomes 0 at first push, incremented/decremented with each push/pop, becomes -1 after pop the final item
-----------------------------------------------------------------------------------------------------------------------------------------------

Usage::

    Tranche tr ; 
    tr.curr = -1 ;


::

    #define TRANCHE_STACK_SIZE 4
    struct Tranche
    {
        float      tmin[TRANCHE_STACK_SIZE] ;  
        unsigned  slice[TRANCHE_STACK_SIZE] ; 
        int curr ;
    };




csg_stack.h : csg.curr starts at -1, becomes 0 at first push, incremented/decremented with each push/pop, becomes -1 after pop the final item
-----------------------------------------------------------------------------------------------------------------------------------------------

Usage::

    CSG_Stack csg ;  
    csg.curr = -1 ;


::

    #define CSG_STACK_SIZE 15
    struct CSG_Stack 
    {
       float4 data[CSG_STACK_SIZE] ; 
       unsigned idx[CSG_STACK_SIZE] ; 
       int curr ;
    };




