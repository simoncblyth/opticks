#include <cstring>
#include "SPath.hh"
#include "SProc.hh"
#include "SGeo.hh"
#include "PLOG.hh"

const plog::Severity SGeo::LEVEL = PLOG::EnvLevel("SGeo", "DEBUG"); 

const char* SGeo::LAST_UPLOAD_CFBASE = nullptr ;

/**
SGeo::SetLastUploadCFBase
---------------------------

Canonically invoked from CSGFoundry::upload with CSGFoundry::getOriginCFBase

HMM: actually when dynamic prim selection has been applied should be minting 
and using a new CFBase for consistency between results and geometry

**/

void SGeo::SetLastUploadCFBase(const char* cfbase)
{
    if(cfbase == nullptr)
    {
        LOG(error) << " cfbase IS NULL : will not be able to save results together with geometry as cfbase not available " ; 
    }
    else
    {
        LOG(error) << " cfbase " << cfbase ;  
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

**/
const char* SGeo::LastUploadCFBase_OutDir()
{
    const char* cfbase = LastUploadCFBase(); 
    if(cfbase == nullptr) return nullptr ; 
    const char* exename = SProc::ExecutableName(); 
    const char* outdir = SPath::Resolve(cfbase, exename, DIRPATH );  
    return outdir ; 
}




