#pragma once
/**
SOPTIX_Module.h
===============

7.5 p48

After using modules to create an OptixPipeline through the OptixProgramGroup
objects, modules may be destroyed with optixModuleDestroy.


**/

#include "SOPTIX_Options.h" 

struct SOPTIX_Module
{ 
    const char* ptxpath ; 
    std::string ptx ; 

    SOPTIX_Options options ; 

    size_t sizeof_log = 0 ;
    char log[2048]; 

    OptixModule module ; 
    
    SOPTIX_Module( OptixDeviceContext& context, const char* _ptxpath ); 
    std::string desc() const ;
};

inline std::string SOPTIX_Module::desc() const 
{
    std::string _log( log, log+sizeof_log ); 
    std::stringstream ss ; 
    ss << "[SOPTIX_Module::desc"  ;
    ss << " ptxpath\n" << ptxpath << "\n" ; 
    ss << " log\n" << _log << "\n" ; 
    ss << " options\n" << options.desc() << "\n" ; 
    ss << "]SOPTIX_Module::desc"  ;
    std::string str = ss.str() ; 
    return str ; 
}


inline SOPTIX_Module::SOPTIX_Module( OptixDeviceContext& context, const char* _ptxpath )
    :
    ptxpath(_ptxpath ? strdup(_ptxpath) : nullptr )
{
    bool read_ok = spath::Read(ptx, ptxpath );
    assert(  read_ok );    

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                context,
                &options.moduleCompileOptions,
                &options.pipelineCompileOptions,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ) );

    if(sizeof_log > 0) std::cout << desc() ; 
    assert( sizeof_log == 0 ); 
}

