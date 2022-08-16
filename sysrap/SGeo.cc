#include <cstring>
#include <sstream>
#include "SPath.hh"
#include "SProc.hh"
#include "SGeo.hh"
#include "SEventConfig.hh"
#include "PLOG.hh"

const plog::Severity SGeo::LEVEL = PLOG::EnvLevel("SGeo", "DEBUG"); 

const char* SGeo::LAST_UPLOAD_CFBASE = nullptr ;

/**
SGeo::SetLastUploadCFBase
---------------------------

Canonically invoked from CSGFoundry::upload with CSGFoundry::getOriginCFBase

HMM: actually when dynamic prim selection has been applied should be minting 
and using a new CFBase for consistency between results and geometry
This is the reason that dynamic prim selection is only appropriate for shortterm
tests such as render scanning for bottlenecks. 

**/

void SGeo::SetLastUploadCFBase(const char* cfbase)
{
    if(cfbase == nullptr)
    {
        LOG(LEVEL) << " cfbase IS NULL : will not be able to save results together with geometry as cfbase not available " ; 
    }
    else
    {
        LOG(LEVEL) << " cfbase " << cfbase ;  
    }
    LAST_UPLOAD_CFBASE = cfbase ? strdup(cfbase) : nullptr ; 
}   
const char* SGeo::LastUploadCFBase() 
{
    return LAST_UPLOAD_CFBASE ; 
}




/**
SGeo::LastUploadCFBase_OutDir
------------------------------

This provides a default output directory to QEvent::save
which is within the last uploaded CFBase/ExeName

HMM: This is becoming less useful due to its fixation on CFBase 
which is not longer the only way to start from a saved geometry. 

**/
const char* SGeo::LastUploadCFBase_OutDir()
{
    const char* cfbase = LastUploadCFBase(); 
    if(cfbase == nullptr) return nullptr ; 
    const char* exename = SProc::ExecutableName(); 
    const char* outdir = SPath::Resolve(cfbase, exename, DIRPATH );  
    return outdir ; 
}


/**
SGeo::DefaultDir
------------------

The FALLBACK_DIR which is used for the SGeo::DefaultDir 
is obtained from SEventConfig::OutFold which is normally "$DefaultOutputDir" $TMP/GEOM/ExecutableName
This can be overriden using SEventConfig::SetOutFold or by setting the 
envvar OPTICKS_OUT_FOLD.

Initially tried to implement FALLBACK_DIR as a static const, but the 
problem with that is that the order of static initialization is not defined,
so having a static const depend on another static const is not a good idea. 

It is normally much easier to use the default of "$DefaultOutputDir" as this 
takes care of lots of the bookkeeping automatically.
However in some circumstances such as with the B side of aligned running (U4RecorderTest) 
it is appropriate to use the override code or envvar to locate B side outputs together 
with the A side. 

**/

const char* SGeo::DefaultDir()
{
    const char* dir_ = LastUploadCFBase_OutDir(); 
    const char* dir = dir_ ? dir_ : SEventConfig::OutFold()  ; 
    if( dir == nullptr ) std::cout << "SGeo::DefaultDir ERR null " << std::endl <<  Desc() ; 

    return dir ; 
}

std::string SGeo::Desc() 
{
    const char* lucfbod = LastUploadCFBase_OutDir() ; 
    const char* outfold = SEventConfig::OutFold() ; 

    std::stringstream ss ; 
    ss << "SGeo::Desc" << std::endl 
       << " SGeo::LastUploadCFBase_OutDir " << ( lucfbod ? lucfbod : "-" ) << std::endl
       << " SEventConfig::OutFold() " << ( outfold ? outfold : "-" ) << std::endl 
       ;

    std::string s = ss.str(); 
    return s ; 
}


