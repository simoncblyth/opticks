#include <iostream>
#include <string>
#include <sstream>

#include <cstdlib>
#include <cstring>

#include "BOpticksResource.hh"
#include "OptiXTest.hh"

#include "PLOG.hh"

std::string OptiXTest::ptxname_( const char* projname, const char* name)
{
   std::stringstream ss ; 
   ss << projname << "_generated_" << name << ".ptx" ; 
   return ss.str();
}

const char* OptiXTest::ptxpath_( const char* cu, const char* projdir, const char* projname)
{
   std::string ptxname = ptxname_(projname, cu) ; 
   std::string ptxpath = BOpticksResource::BuildProduct(projdir, ptxname.c_str());
   return strdup(ptxpath.c_str()) ; 
}

OptiXTest::OptiXTest(optix::Context& context, const char* cu, const char* raygen_name, const char* exception_name)
    :
    m_cu(strdup(cu)),
    m_ptxpath(ptxpath_(cu)),
    m_raygen_name(strdup(raygen_name)),
    m_exception_name(strdup(exception_name))
{
    init(context);
}

void OptiXTest::init(optix::Context& context)
{
    LOG(info) << "OptiXTest::init"
              << description()
               ; 

    context->setEntryPointCount( 1 );

    optix::Program raygenProg    = context->createProgramFromPTXFile(m_ptxpath, m_raygen_name);
    optix::Program exceptionProg = context->createProgramFromPTXFile(m_ptxpath, m_exception_name);

    context->setRayGenerationProgram(0,raygenProg);
    context->setExceptionProgram(0,exceptionProg);
}

std::string OptiXTest::description()
{
    std::stringstream ss ; 
    ss  
              << " cu " << m_cu
              << " ptxpath " << m_ptxpath
              << " raygen " << m_raygen_name 
              << " exception " << m_exception_name 
              ;

    return ss.str(); 
}


void OptiXTest::Summary(const char* msg)
{
    LOG(info) << msg << description() ;
}


