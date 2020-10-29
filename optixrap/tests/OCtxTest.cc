// om-;TEST=OCtxTest om-t
#include "OKConf.hh"
#include "SStr.hh"
#include "OPTICKS_LOG.hh"

#include "OXPPNS.hh"
#include "OCtx.hh"


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

    test_adopt_context_0();  
    test_adopt_context_1();  
 
    return 0 ; 
}
// om-;TEST=OCtxTest om-t
