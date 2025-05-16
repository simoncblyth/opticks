#pragma once
/**
SOPTIX_Module.h : Create OptixModule loaded from file
================================================================

WIP : Generalize this to work with optixir binary as well as PTX text

7.5 p48

After using modules to create an OptixPipeline through the OptixProgramGroup
objects, modules may be destroyed with optixModuleDestroy.


**/

#include "SOPTIX_Options.h" 
#include "spath.h"
#include "sstr.h"

struct SOPTIX_Module
{ 
    enum { UNKNOWN, PTX, OPTIXIR } ; 
    static constexpr const char* UNKNOWN_ = "UNKNOWN" ; 
    static constexpr const char* PTX_     = "PTX" ; 
    static constexpr const char* OPTIXIR_ = "OPTIXIR" ; 
    static const char* Type(int type) ; 

    OptixDeviceContext& context ; 
    const SOPTIX_Options& options ; 
    const char* path ; 
    int type ; 
    std::vector<char> bin ; 
    OptixModule module ; 
    
    std::string desc() const ;

    SOPTIX_Module( 
        OptixDeviceContext& context, 
        const SOPTIX_Options& options,
        const char* _path
        ); 

    void init(); 
};

inline const char* SOPTIX_Module::Type(int type)
{
    const char* str = nullptr ; 
    switch(type)
    {
        case UNKNOWN: str = UNKNOWN_ ; break ; 
        case PTX:     str = PTX_     ; break ; 
        case OPTIXIR: str = OPTIXIR_ ; break ; 
    }
    return str ; 

}

inline std::string SOPTIX_Module::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Module::desc"  ;
    ss << " path\n" << path << "\n" ; 
    ss << " type\n" << type << "\n" ; 
    ss << " Type\n" << Type(type) << "\n" ; 
    ss << " bin.size\n" << bin.size() << "\n" ; 
    ss << " options\n" << options.desc() << "\n" ; 
    ss << "]SOPTIX_Module::desc"  ;
    std::string str = ss.str() ; 
    return str ; 
}


inline SOPTIX_Module::SOPTIX_Module( 
    OptixDeviceContext&   _context, 
    const SOPTIX_Options& _options, 
    const char*           _path
    )
    :
    context(_context),
    options(_options),
    path(spath::Resolve(_path)),
    type(UNKNOWN)
{
    init();
}


/**
SOPTIX_Module::init
---------------------

HMM: in OptiX 7.5 seems the optixModuleCreateFromPTX 
also does optixir ? 

**/

inline void SOPTIX_Module::init()
{
    if(      sstr::EndsWith(path, "ptx") )    type = PTX  ; 
    else if( sstr::EndsWith(path, "optixir")) type = OPTIXIR ; 

    bool read_ok = spath::Read(bin, path );  // should work with PTX as well as OPTIXIR
    std::cout << "SOPTIX_Module::init\n" << desc() << "\n" ; 

    assert(  read_ok );    

    size_t sizeof_log = 0 ;
    char log[2048]; 

#if OPTIX_VERSION <= 70600
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                context,
                &options.moduleCompileOptions,
                &options.pipelineCompileOptions,
                bin.data(),
                bin.size(),
                log,
                &sizeof_log,
                &module
                ) );
#else
    OPTIX_CHECK_LOG( optixModuleCreate(
                context,
                &options.moduleCompileOptions,
                &options.pipelineCompileOptions,
                bin.data(),
                bin.size(),
                log,
                &sizeof_log,
                &module
                ) );

#endif

    std::string _log( log, log+sizeof_log ); 
    if(sizeof_log > 0) std::cout << _log ; 
    assert( sizeof_log == 0 ); 
}

