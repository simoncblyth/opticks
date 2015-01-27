#include "OptiXProgram.hh"

#include <sstream>

std::string generated_ptx_path( const char* folder, const char* target, const char* base )
{
  std::stringstream ss;
  ss << folder << "/" << target << "_generated_" << base << ".ptx" ;
  return ss.str() ;
}


OptiXProgram::OptiXProgram(const char* ptxfold, const char* target )
        : 
        m_context(NULL),
        m_ptxfold(NULL),
        m_target(NULL)
{
    if(!ptxfold || !target ) return ; 
    printf("OptiXProgram::OptiXProgram ctor ptxfold %s target %s  \n", ptxfold, target );

    m_ptxfold = strdup(ptxfold);
    m_target = strdup(target);
}

OptiXProgram::~OptiXProgram(void)
{
    printf("OptiXProgram dtor\n");

    free(m_ptxfold);
    free(m_target);
}


void OptiXProgram::setContext(optix::Context& context)
{
     m_context = context ; 
}


const char* const OptiXProgram::ptxpath( const std::string& base )
{
  static std::string path;
  path = generated_ptx_path(m_ptxfold, m_target, base.c_str());
  return path.c_str();
}


optix::Program OptiXProgram::createProgram(const char* filename, const char* fname )
{
  std::string path = ptxpath(filename); 
  std::string key = path + ":" + fname ; 

  if(m_programs.find(key) == m_programs.end())
  { 
       printf("createProgram key %s \n", key.c_str() );
       optix::Program program = m_context->createProgramFromPTXFile( path.c_str(), fname ); 
       m_programs[key] = program ; 
  } 
  else
  {
       //printf("createProgram cached key %s \n", key.c_str() );
  } 
  return m_programs[key];
}


