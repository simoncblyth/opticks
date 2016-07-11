#include "MultiViewNPY.hpp"
#include "AxisApp.hh"

#include "OGLRAP_LOG.hh"
#include "PLOG.hh"







#include "OXRAP_PUSH.hh"
#include <optixu/optixpp_namespace.h>
#include "OXRAP_POP.hh"

#include "BOpticksResource.hh"
#include "NPY.hpp"

#include <iostream>
#include <string>
#include <sstream>

#include <cstdlib>
#include <cstring>


std::string ptxname_( const char* target, const char* name)
{
   std::stringstream ss ; 
   ss << target << "_generated_" << name << ".ptx" ; 
   return ss.str();
}

std::string ptxpath_( const char* proj, const char* target, const char* name)
{
   std::string ptxname = ptxname_(target, name) ; 
   std::string ptxpath = BOpticksResource::BuildProduct(proj, ptxname.c_str());
   return ptxpath ; 
}



/*
class AxisModify {
    public:
        AxisModify(optix::Context& context);
    private:
        void init();
    private:
        optix::Context m_context ; 
}
*/



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    OGLRAP_LOG__ ; 

    LOG(info) << argv[0] ; 

    AxisApp aa(argc, argv); 

    MultiViewNPY* attr = aa.getAxisAttr();

    try {

        optix::Context context = optix::Context::create();
        context->setEntryPointCount( 1 );

        unsigned width = 512 ; 
        unsigned height = 512 ; 
     
        optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
        context["output_buffer"]->set(buffer);

        std::string ptxpath = ptxpath_("optixrap", "OptiXRap", "minimalTest.cu" ) ;
        std::cerr << " ptxpath " << ptxpath << std::endl ; 

        optix::Program raygenProg    = context->createProgramFromPTXFile(ptxpath.c_str(), "minimal");
        optix::Program exceptionProg = context->createProgramFromPTXFile(ptxpath.c_str(), "exception");

        context->setRayGenerationProgram(0,raygenProg);
        context->setExceptionProgram(0,exceptionProg);

        context->validate();
        context->compile();
        context->launch(0, width, height);


        NPY<float>* npy = NPY<float>::make(width, height, 4) ;
        npy->zero();

        void* ptr = buffer->map() ; 
        npy->read( ptr );
        buffer->unmap(); 

    } 
    catch( optix::Exception& e )
    {
        std::cerr <<  e.getErrorString().c_str() << std::endl ; 
        exit(1);
    }










    aa.renderLoop();

    return 0 ; 
}



