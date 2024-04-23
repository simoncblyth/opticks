#pragma once
/**
SOPTIX_Module.h : Create OptixModule from PTX loaded from file
================================================================

7.5 p48

After using modules to create an OptixPipeline through the OptixProgramGroup
objects, modules may be destroyed with optixModuleDestroy.


**/

#include "SOPTIX_Options.h" 

struct SOPTIX_Module
{ 
    OptixDeviceContext& context ; 
    const SOPTIX_Options& options ; 
    const char* ptxpath ; 

    std::string ptx ; 
    OptixModule module ; 
    
    std::string desc() const ;

    SOPTIX_Module( 
        OptixDeviceContext& context, 
        const SOPTIX_Options& options,
        const char* _ptxpath
        ); 

    void init(); 
};

inline std::string SOPTIX_Module::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Module::desc"  ;
    ss << " ptxpath\n" << ptxpath << "\n" ; 
    ss << " options\n" << options.desc() << "\n" ; 
    ss << "]SOPTIX_Module::desc"  ;
    std::string str = ss.str() ; 
    return str ; 
}


inline SOPTIX_Module::SOPTIX_Module( 
    OptixDeviceContext&   _context, 
    const SOPTIX_Options& _options, 
    const char*           _ptxpath
    )
    :
    context(_context),
    options(_options),
    ptxpath(_ptxpath ? strdup(_ptxpath) : nullptr )
{
    init();
}

inline void SOPTIX_Module::init()
{
    bool read_ok = spath::Read(ptx, ptxpath );
    assert(  read_ok );    

    size_t sizeof_log = 0 ;
    char log[2048]; 

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

    std::string _log( log, log+sizeof_log ); 
    if(sizeof_log > 0) std::cout << _log ; 
    assert( sizeof_log == 0 ); 
}

