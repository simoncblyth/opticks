// https://devtalk.nvidia.com/default/topic/734914/optix/optix-bug-crash-with-cuda-error-kernel-ret-700-when-not-rtprinting-anything-small-demo-code-/

#include "OXRAP_PUSH.hh"
#include <optixu/optixpp_namespace.h>
#include "OXRAP_POP.hh"

#include "NPY.hpp"

#include <iostream>
#include <string>
#include <sstream>

#include <cstdlib>
#include <cstring>


std::string ptxpath_( const char* target, const char* name)
{
   std::stringstream ss ; 
   ss << getenv("HOME") << "/local/opticks/build/optixrap/" << target << "_generated_" << name << ".ptx" ; 
   return ss.str();
}


int main( int , char** argv ) {

  std::cout << argv[0] << std::endl ;  

  try {

    optix::Context context = optix::Context::create();
    context->setEntryPointCount( 1 );

    unsigned width = 512 ; 
    unsigned height = 512 ; 
 
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    context["output_buffer"]->set(buffer);

    std::string ptxpath = ptxpath_("OptiXRap", "minimalTest.cu" ) ;
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

    npy->save("$TMP/OOMinimalTest.npy");


  } 
  catch( optix::Exception& e )
  {
      std::cerr <<  e.getErrorString().c_str() << std::endl ; 
      exit(1);
  }

  return 0;
}
