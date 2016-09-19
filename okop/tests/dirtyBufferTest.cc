#include "OContext.hh"
#include "OBuf.hh"
#include "TBuf.hh"
#include "OXPPNS.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    optix::Context context = optix::Context::create();

    optix::Buffer test_buffer = context->createBuffer( RT_BUFFER_INPUT );

    test_buffer->setFormat(RT_FORMAT_UNSIGNED_INT);

    unsigned size = 100 ; 

    NPT<unsigned>* buf = NPY<unsigned>::make(size) ;
    buf->fill(42);
    buf->dump();


    test_buffer->setSize(size);

    context["test_buffer"]->setBuffer(test_buffer);  

    bool with_top = false ; 

    OContext ctx(context, OContext::COMPUTE, with_top);

    int entry = ctx.addEntry("dirtyBufferTest.cu.ptx", "dirtyBufferTest", "exception");

    ctx.launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, NULL);

    ctx.launch( OContext::LAUNCH, entry,  size, 1, NULL ); 



    return 0 ; 
}
