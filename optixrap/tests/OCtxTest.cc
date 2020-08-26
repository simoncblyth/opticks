#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "NPY.hpp"
#include "OCtx.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    NPY<float>* arr = NPY<float>::make(10, 4) ; 

    LOG(info) << " arr " << arr->getShapeString() ;  
    
    OCtx_create_buffer(arr, "some_buffer", 'O', ' '); 


    return 0 ; 
}
