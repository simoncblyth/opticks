/*
optixtest()
{
    local nam=cbrtTest
    local exe=/tmp/$nam
    local ptx=/tmp/$nam.ptx

    local cc=../tests/$nam.cc
    local cu=$nam.cu

    local ver=OptiX_380
    local inc=/Developer/$ver/include 
    local lib=/Developer/$ver/lib64 

    clang -I/usr/local/cuda/include -I$inc -L$lib -loptix  -lc++  -Wl,-rpath,$lib  $cc  -o $exe

    nvcc -arch=sm_30 -ptx $cu -I$inc -o $ptx

    $exe $ptx $nam
}

optixtest cbrtTest 
optixtest intersect_analytic_test


*/

#include <optixu/optixpp_namespace.h>
#include <string>

struct OptiXMinimalTest 
{
   const char* m_ptxpath ; 
   const char* m_raygen_name ; 
   const char* m_exception_name ; 
     
   OptiXMinimalTest(optix::Context& context, const char* ptxpath, const char* raygen_name, const char* exception_name="exception"); 
   std::string description();

   void init(optix::Context& context);  
};


#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cassert>

OptiXMinimalTest::OptiXMinimalTest(optix::Context& context, const char* ptxpath, const char* raygen_name, const char* exception_name)
    :
    m_ptxpath(strdup(ptxpath)),
    m_raygen_name(strdup(raygen_name)),
    m_exception_name(strdup(exception_name))
{
    init(context);
}

void OptiXMinimalTest::init(optix::Context& context)
{
    context->setEntryPointCount( 1 );

    optix::Program raygenProg    = context->createProgramFromPTXFile(m_ptxpath, m_raygen_name);
    optix::Program exceptionProg = context->createProgramFromPTXFile(m_ptxpath, m_exception_name);

    context->setRayGenerationProgram(0,raygenProg);
    context->setExceptionProgram(0,exceptionProg);

    context->setPrintEnabled(true);
    context->setPrintBufferSize(2*2*2*8192);

}

std::string OptiXMinimalTest::description()
{
    std::stringstream ss ; 
    ss  
              << " ptxpath " << m_ptxpath
              << " raygen " << m_raygen_name 
              << " exception " << m_exception_name 
              ;

    return ss.str(); 
}


