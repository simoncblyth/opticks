#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "NPY.hpp"
#include "OCtx.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    const NPY<float>* arr = NPY<float>::make(10, 4) ; 

    LOG(info) << " arr " << arr->getShapeString() ;  
    
    int item = -1 ; 
    const char* key = "some_context_key" ; 
    char type = 'O' ;  
    char flag = ' ' ; 

    OCtx::Get()->create_buffer(arr, key, type, flag, item); 


    return 0 ; 
}
