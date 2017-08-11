#include "OptiXMinimalTest.hh"


int main( int argc, char** argv ) 
{
    optix::Context context = optix::Context::create();

    RTsize stack_size_bytes = context->getStackSize() ;
    //stack_size_bytes *= 2 ; 
    //context->setStackSize(stack_size_bytes);
   
    OptiXMinimalTest* test = new OptiXMinimalTest(context, argc, argv  ) ;
    std::cout << test->description() << std::endl ; 

    //unsigned width = 512 ; 
    //unsigned height = 512 ; 

    unsigned width = 1 ; 
    unsigned height = 1 ; 

    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);

    context->validate();
    context->compile();
    context->launch(0, width, height);

    return 0;
}
