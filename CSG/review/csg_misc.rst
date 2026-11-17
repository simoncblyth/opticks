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



::

    __global__ void processSlices(ushort2* d_slices, float* d_data) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Loads exactly 32 bits from global memory in one clean instruction
        ushort2 slice = d_slices[tid];

        ushort start = slice.x;
        ushort end   = slice.y;

        // Use them directly
        for (int i = start; i < end; ++i) {
            // ... process data ...
        }
    }


Because you are using these 16-bit values for slicing (which implies array
indexing and loop bounds), be aware that loop counters should ideally be 32-bit
signed integers (int).

When you assign slice.x or slice.y to a loop variable, implicitly or explicitly
cast it to an int so that the GPU can run the loop math natively at 32-bit
speed without 16-bit overflow tracking overhead::


    // Recommended: Upcast to standard int for the actual loop execution
    int start = slice.x;
    int end   = slice.y;

    for (int i = start; i < end; ++i) {
        // Native 32-bit loop math
    }









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




