#ifndef OPTIXPROGRAM_H
#define OPTIXPROGRAM_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <string>
#include <map>


class OptiXProgram  
{
public:
    OptiXProgram(const char* ptxfold, const char* target);

    virtual ~OptiXProgram();

    const char* const ptxpath( const std::string& filename );

public:
    optix::Program createProgram(const char* filename, const char* fname );

    void setContext(optix::Context& context);

private:

    char* m_ptxfold ; 

    char* m_target ; 

private:

    optix::Context m_context ;

    std::map<std::string,optix::Program> m_programs;

};


#endif

