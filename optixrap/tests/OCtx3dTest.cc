// om-;TEST=OCtx3dTest om-t

#include "OKConf.hh"
#include "SStr.hh"
#include "OPTICKS_LOG.hh"

#include "OXPPNS.hh"
#include "NPY.hpp"
#include "OCtx.hh"

const char* CMAKE_TARGET = "OCtx3dTest" ; 
const char* PTXPATH = OKConf::PTXPath(CMAKE_TARGET, SStr::Concat(CMAKE_TARGET, ".cu" ), "tests" );      

void check_array_3d( const NPY<int>* a, bool transpose_buffer );
void dump_array_3d( const NPY<int>* a, bool transpose_buffer );


/**
test_populate_3d_buffer
------------------------

There are 3 shapes to consider:

1. NPY array (row-major, see npy/tests/NPY5Test.cc:test_row_major_serialization)
2. OptiX launch 
3. OptiX buffer (column-major by observation)

When filling the buffer with launch_index indexing the launch and buffer dimensions must match,
otherwise get indexing errors when exceptions are enabled::

    rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
    rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

    rtBuffer<int4,3> buffer;

    RT_PROGRAM void raygen()
    {
        buffer[launch_index] = make_int4(launch_index.x, launch_index.y, launch_index.z, launch_dim.z ) ; 
    }

**/

void test_populate_3d_buffer(bool transpose_buffer)
{
    OCtx* octx = OCtx::Get() ;  // reuse any preexisting context 

    int ni = 2 ;    // depth 
    int nj = 8 ;    // height
    int nk = 16 ;   // width 
    int nl = 4 ;    // <-- no name for last dimension, it collapses to form the element

    NPY<int>* arr = NPY<int>::make(ni, nj, nk, nl) ; 
    
    const char* key = "buffer" ; 
    char type = 'O' ;  // 'O': output 
    char flag = ' ' ;  // ' ':  default  
    int item = -1 ;    //  -1:  whole array maps to the buffer

    octx->create_buffer(arr, key, type, flag, item, transpose_buffer ); 

    unsigned entry_point_index = 0u ; 
    octx->set_raygen_program(    entry_point_index, PTXPATH, "raygen" );
    octx->set_exception_program( entry_point_index, PTXPATH, "exception" );

    unsigned l0 = transpose_buffer ? nk : ni ;  
    unsigned l1 = transpose_buffer ? nj : nj ;  
    unsigned l2 = transpose_buffer ? ni : nk ;  

    std::cout << " launch_dimensions (l0,l1,l2) (" << l0 << "," << l1 << "," << l2 << ")" << std::endl ; 
    octx->launch( entry_point_index, l0, l1, l2 );

    arr->zero();
    octx->download_buffer(arr, key, item);
   
    check_array_3d(arr, transpose_buffer); 
    dump_array_3d(arr, transpose_buffer); 

    const char* path = SStr::Concat("$TMP/optixrap/tests/OCtx3dTest/test_populate_3d_buffer/arr_", int(transpose_buffer), ".npy") ; 
    LOG(info) << " saving to " << path ; 
    arr->save(path); 
}


void check_array_3d( const NPY<int>* a, bool transpose_buffer )
{
    int ni = a->getShape(0); 
    int nj = a->getShape(1); 
    int nk = a->getShape(2); 

    int nl = a->getShape(3); 
    assert( nl == 4 ); 

    for(int i=0 ; i < ni ; i++) 
    for(int j=0 ; j < nj ; j++) 
    for(int k=0 ; k < nk ; k++) 
    {
        glm::ivec4 q = a->getQuad_(i, j, k);
        if( transpose_buffer == true )
        {
            assert( q.x == k ); 
            assert( q.y == j ); 
            assert( q.z == i ); 
            assert( q.w == ni ); 
        }
        else
        {
           // when not transposing have a fold-over "mess"
            assert( q.w == nk ); 
        } 
    }
}

void dump_array_3d( const NPY<int>* a, bool transpose_buffer )
{
    std::cout << " transpose_buffer " << transpose_buffer << ( transpose_buffer ? " -> OK" : " -> SCRAMBLED MESS " ) << std::endl ; 
    std::cout << " a.shape " << a->getShapeString() << std::endl ; 
 
    int ni = a->getShape(0);  // depth
    int nj = a->getShape(1);  // height 
    int nk = a->getShape(2);  // width 

    int nl = a->getShape(3); 
    assert( nl == 4 ); 

    for(int i=0 ; i < ni ; i++)           // depth index
    {
        std::cout << std::endl << "(i:" << std::setw(1) << i << ") " << std::endl ; 

        std::cout << std::setw(8) << "(k,j,i)" ; 
        for(int k=0 ; k < nk ; k++) std::cout << "(k:" << std::setw(1) << std::hex << k << ") " ; 
        std::cout << std::endl ;

        for(int j=0 ; j < nj ; j++)      // height
        {
            std::cout << "(j:" << std::setw(1) << j << ")  " ; 

            for(int k=0 ; k < nk ; k++)      // width  
            {

               glm::ivec4 q = a->getQuad_(i, j, k ); 

               std::stringstream ss ; 
               ss
                   << std::setw(1) << std::hex << q.x
                   << ","  
                   << std::setw(1) << std::hex << q.y
                   << ","
                   << std::setw(1) << std::hex << q.z
                    ;

               std::cout << std::setw(6) << ss.str() ;  

               if( k == nk - 1)
                   std::cout 
                      << " ("
                      << std::setw(2) << std::hex << q.w
                      << ") "
                      ;

            }
            std::cout << std::endl ; 
        }
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_populate_3d_buffer(true); 
    test_populate_3d_buffer(false); 
 
    return 0 ; 
}
// om-;TEST=OCtx3dTest om-t
