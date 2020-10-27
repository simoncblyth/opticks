#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "NPY.hpp"
#include "OCtx.hh"

void test_create_buffer()
{
    const NPY<float>* arr = NPY<float>::make(10, 4) ; 

    LOG(info) << " arr " << arr->getShapeString() ;  
    
    int item = -1 ; 
    const char* key = "some_context_key" ; 
    char type = 'O' ;  
    char flag = ' ' ; 

    OCtx::Get()->create_buffer(arr, key, type, flag, item); 
}


void test_adopt_context_0()
{
   optix::Context context = optix::Context::create(); 

   optix::ContextObj* contextObj = context.get(); 
   RTcontext context_ptr = contextObj->get(); 
   void* ptr = (void*)context_ptr ; 

   OCtx octx(ptr); 
}

void test_adopt_context_1()
{
    optix::Context context = optix::Context::create(); 
    OCtx octx((void*)(context.get()->get())); 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_create_buffer(); 
    //test_adopt_context_0();  
    test_adopt_context_1();  

 
    return 0 ; 
}
// om-;TEST=OCtxTest om-t
