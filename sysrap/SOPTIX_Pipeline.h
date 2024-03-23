#pragma once
/**
SOPTIX_Pipeline.h
==================

**/

struct SOPTIX_Pipeline
{ 
    static constexpr const char* RG = "__raygen__rg" ;
    static constexpr const char* MS = "__miss__ms" ;
    static constexpr const char* IS = "__intersection__is" ;
    static constexpr const char* CH = "__closesthit__ch" ;
    static constexpr const char* AH = "__anyhit__ah" ;

    OptixDeviceContext& context ; 
    OptixModule& module ; 
    const SOPTIX_Options& options ;  

    OptixProgramGroup raygen_pg   = nullptr;
    OptixProgramGroup miss_pg     = nullptr;
    OptixProgramGroup hitgroup_pg = nullptr;

    OptixPipeline pipeline = nullptr;

    OptixStackSizes stackSizes = {}; 

    uint32_t max_trace_depth = 1;   // only RG invokes trace, no recursion   
    uint32_t max_cc_depth = 0;  
    uint32_t max_dc_depth = 0;  

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;

    // see optix7-;optix7-host : it states that IAS->GAS needs to be two  
    unsigned maxTraversableGraphDepth = 2 ; 



    std::string desc() const ;

    SOPTIX_Pipeline( 
        OptixDeviceContext& context, 
        OptixModule& module,
        const SOPTIX_Options& options 
        ); 

    void init();
    void initRaygen(); 
    void initMiss(); 
    void initHitgroup();
    void initPipeline();

    std::string descStack() const ;
    static std::string DescStackSizes(const OptixStackSizes& stackSizes ); 
 
};


inline std::string SOPTIX_Pipeline::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Pipeline::desc"  ;
    ss << " options\n" << options.desc() << "\n" ; 
    ss << " pipeline " << ( pipeline ? "YES" : "NO " ) << "\n" ; 
    ss << descStack() << "\n" ;
    ss << "]SOPTIX_Pipeline::desc"  ;
    std::string str = ss.str() ; 
    return str ; 
}



inline SOPTIX_Pipeline::SOPTIX_Pipeline( 
    OptixDeviceContext& _context, 
    OptixModule& _module, 
    const SOPTIX_Options& _options 
    )
    :
    context(_context),
    module(_module),
    options(_options)
{
    init();  
}

inline void SOPTIX_Pipeline::init()
{
    initRaygen();
    initMiss();
    initHitgroup();
    initPipeline();
}


inline void SOPTIX_Pipeline::initRaygen()
{
    OptixProgramGroupKind kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN ;
    OptixProgramGroupFlags flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE ; 

    OptixProgramGroupDesc desc = {};
    desc.kind = kind ;
    desc.flags = flags ;  
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = RG ;

    size_t sizeof_log = 0 ;
    char log[2048];
    unsigned num_program_groups = 1 ;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                context,
                &desc,
                num_program_groups,
                &(options.programGroupOptions),
                log,
                &sizeof_log,
                &raygen_pg
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ;
    assert( sizeof_log == 0);
}

inline void SOPTIX_Pipeline::initMiss()
{
    OptixProgramGroupKind kind = OPTIX_PROGRAM_GROUP_KIND_MISS ;
    OptixProgramGroupFlags flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE ; 

    OptixProgramGroupDesc desc = {};
    desc.kind = kind ;
    desc.flags = flags ;  
    desc.miss.module = module;
    desc.miss.entryFunctionName = MS ;

    size_t sizeof_log = 0 ;
    char log[2048];
    unsigned num_program_groups = 1 ;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                context,
                &desc,
                num_program_groups,
                &(options.programGroupOptions),
                log,
                &sizeof_log,
                &miss_pg
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ;
    assert( sizeof_log == 0);
}


/**
SOPTIX_Pipeline::initHitgroup
--------------------------------

HMM: no IS for tri ? 

**/

inline void SOPTIX_Pipeline::initHitgroup()
{
    OptixProgramGroupKind kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP  ;
    OptixProgramGroupFlags flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE ; 
    OptixProgramGroupDesc desc = {};
    desc.kind = kind ; 
    desc.flags = flags ;  

    desc.hitgroup.moduleCH            = module  ;
    desc.hitgroup.entryFunctionNameCH = CH ;

    desc.hitgroup.moduleAH            = nullptr ;
    desc.hitgroup.entryFunctionNameAH = nullptr ;

    desc.hitgroup.moduleIS            = nullptr ;
    desc.hitgroup.entryFunctionNameIS = nullptr ;


    size_t sizeof_log = 0 ;
    char log[2048];
    unsigned num_program_groups = 1 ;


    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                context,
                &desc,
                num_program_groups,
                &(options.programGroupOptions),
                log,
                &sizeof_log,
                &hitgroup_pg
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ;
    assert( sizeof_log == 0);
    
}

inline void SOPTIX_Pipeline::initPipeline()
{
    OptixProgramGroup program_groups[] = { raygen_pg, miss_pg, hitgroup_pg };
    unsigned num_program_groups = sizeof( program_groups ) / sizeof( program_groups[0] ) ;

    size_t sizeof_log = 0 ;
    char log[2048];

    OPTIX_CHECK_LOG( optixPipelineCreate(
                context,
                &(options.pipelineCompileOptions),
                &(options.pipelineLinkOptions),
                program_groups,
                num_program_groups,
                log,
                &sizeof_log,
                &pipeline
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ;
    assert( sizeof_log == 0);


    for(int i=0 ; i < num_program_groups ; i++)
    {
        OptixProgramGroup& pg = program_groups[i] ; 
        OPTIX_CHECK( optixUtilAccumulateStackSizes( pg, &stackSizes ) );
    }

    OPTIX_CHECK( optixUtilComputeStackSizes(
                &stackSizes,
                max_trace_depth,
                max_cc_depth,
                max_dc_depth,
                &directCallableStackSizeFromTraversal,
                &directCallableStackSizeFromState,
                &continuationStackSize
                ) );


    OPTIX_CHECK( optixPipelineSetStackSize( pipeline,
                                       directCallableStackSizeFromTraversal,
                                       directCallableStackSizeFromState,
                                       continuationStackSize,
                                       maxTraversableGraphDepth ) ) ;
}



std::string SOPTIX_Pipeline::descStack() const
{
    std::stringstream ss ; 
    ss << "SOPTIX_Pipeline::descStack"
       << std::endl 
       << "(inputs to optixUtilComputeStackSizes)" 
       << std::endl 
       << " max_trace_depth " << max_trace_depth
       << " max_cc_depth " << max_cc_depth
       << " max_dc_depth " << max_dc_depth
       << std::endl 
       << " program_group stackSizes "
       << DescStackSizes(stackSizes)
       << "(outputs from optixUtilComputeStackSizes) "
       << std::endl
       << " directCallableStackSizeFromTraversal " << directCallableStackSizeFromTraversal
       << std::endl
       << " directCallableStackSizeFromState " << directCallableStackSizeFromState
       << std::endl
       << " continuationStackSize " << continuationStackSize
       << std::endl
       << "(further inputs to optixPipelineSetStackSize)"
       << std::endl 
       << " maxTraversableGraphDepth " << maxTraversableGraphDepth
       << std::endl 
       ;  
   
    std::string str = ss.str() ; 
    return str ; 
}


std::string SOPTIX_Pipeline::DescStackSizes(const OptixStackSizes& stackSizes ) // static
{
    std::stringstream ss ; 
    ss 
        << " stackSizes.cssRG " << stackSizes.cssRG << "\n"
        << " stackSizes.cssMS " << stackSizes.cssMS << "\n"
        << " stackSizes.cssCH " << stackSizes.cssCH << "\n"
        << " stackSizes.cssAH " << stackSizes.cssAH << "\n"
        << " stackSizes.cssIS " << stackSizes.cssIS << "\n"
        << " stackSizes.cssCC " << stackSizes.cssCC << "\n"
        << " stackSizes.dssDC " << stackSizes.dssDC << "\n"
        ;
    std::string str = ss.str() ; 
    return str ; 
}

