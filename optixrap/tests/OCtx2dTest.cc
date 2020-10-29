// om-;TEST=OCtx2dTest om-t

#include "OKConf.hh"
#include "SStr.hh"
#include "OPTICKS_LOG.hh"

#include "OXPPNS.hh"
#include "NPY.hpp"
#include "OCtx.hh"

const char* CMAKE_TARGET = "OCtx2dTest" ; 
const char* PTXPATH = OKConf::PTXPath(CMAKE_TARGET, SStr::Concat(CMAKE_TARGET, ".cu" ), "tests" );      

void check_array( const NPY<int>* a, bool transpose_buffer );
void dump_array( const NPY<int>* a, bool transpose_buffer );


/**
test_populate_2d_buffer
------------------------

There are 3 shapes to consider:

1. NPY array (row-major, see npy/tests/NPY5Test.cc:test_row_major_serialization)
2. OptiX launch 
3. OptiX buffer (column-major by observation)

When filling the buffer with launch_index indexing the launch and buffer dimensions must match,
otherwise get indexing errors when exceptions are enabled::

    rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
    rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

    rtBuffer<int4,2> buffer;

    RT_PROGRAM void raygen()
    {
        buffer[launch_index] = make_int4(launch_index.x, launch_index.y, launch_dim.x, launch_dim.y ) ; 
    }

Indexing errors look like::

    Caught RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS
      launch index   : 316, 21, 0
      buffer address : 0xFFFDFFFFFFFFFFFF
      dimensionality : 3
      size           : 360x180x1
      element size   : 4
      accessed index : 0, 21, 316



BUT find that with *transpose_buffer = false* the array populated from the buffer is "folded" over, 
because the NPY array deserialization gets a column of 8 where it expects to get a row of 16. This indicates that
OptiX 2d buffers are using column-major serialization::

     launch_dimensions (li,lj) (8,16)
     transpose_buffer 0
     a.shape 8,16,4
           (j:0) (j:1) (j:2) (j:3) (j:4) (j:5) (j:6) (j:7) (j:8) (j:9) (j:a) (j:b) (j:c) (j:d) (j:e) (j:f) 
    (i:0)   0,0   1,0   2,0   3,0   4,0   5,0   6,0   7,0   0,1   1,1   2,1   3,1   4,1   5,1   6,1   7,1  ( 8,16) 
    (i:1)   0,2   1,2   2,2   3,2   4,2   5,2   6,2   7,2   0,3   1,3   2,3   3,3   4,3   5,3   6,3   7,3  ( 8,16) 
    (i:2)   0,4   1,4   2,4   3,4   4,4   5,4   6,4   7,4   0,5   1,5   2,5   3,5   4,5   5,5   6,5   7,5  ( 8,16) 
    (i:3)   0,6   1,6   2,6   3,6   4,6   5,6   6,6   7,6   0,7   1,7   2,7   3,7   4,7   5,7   6,7   7,7  ( 8,16) 
    (i:4)   0,8   1,8   2,8   3,8   4,8   5,8   6,8   7,8   0,9   1,9   2,9   3,9   4,9   5,9   6,9   7,9  ( 8,16) 
    (i:5)   0,a   1,a   2,a   3,a   4,a   5,a   6,a   7,a   0,b   1,b   2,b   3,b   4,b   5,b   6,b   7,b  ( 8,16) 
    (i:6)   0,c   1,c   2,c   3,c   4,c   5,c   6,c   7,c   0,d   1,d   2,d   3,d   4,d   5,d   6,d   7,d  ( 8,16) 
    (i:7)   0,e   1,e   2,e   3,e   4,e   5,e   6,e   7,e   0,f   1,f   2,f   3,f   4,f   5,f   6,f   7,f  ( 8,16) 


With *transpose_buffer = true* get the expected correspondence between buffer and array, but with the added  
cognitive burden of having to transpose the buffer (and launch shape) relative to the array.

     launch_dimensions (li,lj) (16,8)
     transpose_buffer 1
     a.shape 8,16,4
           (j:0) (j:1) (j:2) (j:3) (j:4) (j:5) (j:6) (j:7) (j:8) (j:9) (j:a) (j:b) (j:c) (j:d) (j:e) (j:f) 
    (i:0)   0,0   1,0   2,0   3,0   4,0   5,0   6,0   7,0   8,0   9,0   a,0   b,0   c,0   d,0   e,0   f,0  (16, 8) 
    (i:1)   0,1   1,1   2,1   3,1   4,1   5,1   6,1   7,1   8,1   9,1   a,1   b,1   c,1   d,1   e,1   f,1  (16, 8) 
    (i:2)   0,2   1,2   2,2   3,2   4,2   5,2   6,2   7,2   8,2   9,2   a,2   b,2   c,2   d,2   e,2   f,2  (16, 8) 
    (i:3)   0,3   1,3   2,3   3,3   4,3   5,3   6,3   7,3   8,3   9,3   a,3   b,3   c,3   d,3   e,3   f,3  (16, 8) 
    (i:4)   0,4   1,4   2,4   3,4   4,4   5,4   6,4   7,4   8,4   9,4   a,4   b,4   c,4   d,4   e,4   f,4  (16, 8) 
    (i:5)   0,5   1,5   2,5   3,5   4,5   5,5   6,5   7,5   8,5   9,5   a,5   b,5   c,5   d,5   e,5   f,5  (16, 8) 
    (i:6)   0,6   1,6   2,6   3,6   4,6   5,6   6,6   7,6   8,6   9,6   a,6   b,6   c,6   d,6   e,6   f,6  (16, 8) 
    (i:7)   0,7   1,7   2,7   3,7   4,7   5,7   6,7   7,7   8,7   9,7   a,7   b,7   c,7   d,7   e,7   f,7  (16, 8) 


**/

void test_populate_2d_buffer(bool transpose_buffer)
{
    OCtx* octx = OCtx::Get() ;  // reuse any preexisting context 

    int ni = 8 ;    // height
    int nj = 16 ;   // width 
    int nk = 4 ;    // <-- no name for last dimension, it controls the element type

    NPY<int>* arr = NPY<int>::make(ni, nj, nk) ; 
    
    int item = -1 ; 
    const char* key = "buffer" ; 
    char type = 'O' ;  
    char flag = ' ' ;    

    octx->create_buffer(arr, key, type, flag, item, transpose_buffer ); 

    unsigned entry_point_index = 0u ; 
    octx->set_raygen_program(    entry_point_index, PTXPATH, "raygen" );
    octx->set_exception_program( entry_point_index, PTXPATH, "exception" );

    unsigned li = transpose_buffer ? nj : ni ;  
    unsigned lj = transpose_buffer ? ni : nj ;  

    std::cout << " launch_dimensions (li,lj) (" << li << "," << lj << ")" << std::endl ; 
    octx->launch( entry_point_index, li, lj );

    arr->zero();
    octx->download_buffer(arr, key, -1);
   
    check_array(arr, transpose_buffer); 
    dump_array(arr, transpose_buffer); 

    const char* path = SStr::Concat("$TMP/optixrap/tests/OCtx2dTest/test_populate_2d_buffer/arr_", int(transpose_buffer), ".npy") ; 
    LOG(info) << " saving to " << path ; 
    arr->save(path); 
}


void check_array( const NPY<int>* a, bool transpose_buffer )
{
    int ni = a->getShape(0); 
    int nj = a->getShape(1); 
    int nk = a->getShape(2); 
    assert( nk == 4 ); 

    for(int i=0 ; i < ni ; i++) 
    for(int j=0 ; j < nj ; j++) 
    {
        glm::ivec4 q = a->getQuad_(i, j);
        if( transpose_buffer == true )
        {
            assert( q.x == j ); 
            assert( q.y == i ); 
            assert( q.z == nj ); 
            assert( q.w == ni ); 
        }
        else
        {
           // when not transposing have a fold-over "mess"
            assert( q.z == ni ); 
            assert( q.w == nj ); 
        } 
    }
}

void dump_array( const NPY<int>* a, bool transpose_buffer )
{
    std::cout << " transpose_buffer " << transpose_buffer << std::endl ; 
    std::cout << " a.shape " << a->getShapeString() << std::endl ; 
 
    int ni = a->getShape(0); 
    int nj = a->getShape(1); 
    int nk = a->getShape(2); 

    assert( nk == 4 ); 

    std::cout << std::setw(7) << " " ; 
    for(int j=0 ; j < nj ; j++)
    {
        std::cout 
            << "(j:" << std::setw(1) << std::hex << j << ") "
            ;
    }     
    std::cout << std::endl ;


    for(int i=0 ; i < ni ; i++)           // height index
    {
        std::cout 
            << "(i:" << std::setw(1) << std::hex << i << ") " 
            ; 

        for(int j=0 ; j < nj ; j++){      // width index 

           glm::ivec4 q = a->getQuad_(i, j ); 

           std::stringstream ss ; 
           ss
               << std::setw(1) << std::hex << q.x
               << ","  
               << std::setw(1) << std::hex << q.y
               << " "
                ;

           std::cout << std::setw(6) << ss.str() ;  

           if( j == nj - 1)
               std::cout 
                  << " ("
                  << std::setw(2) << std::dec << q.z
                  << ","  
                  << std::setw(2) << std::dec << q.w
                  << ") "
                  ;

        }
        std::cout << std::endl ; 
    }
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_populate_2d_buffer(true); 
    test_populate_2d_buffer(false); 

 
    return 0 ; 
}
// om-;TEST=OCtx2dTest om-t
