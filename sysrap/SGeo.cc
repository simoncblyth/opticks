#include <cstring>
#include "SPath.hh"
#include "SProc.hh"
#include "SGeo.hh"

const char* SGeo::LAST_UPLOAD_CFBASE = nullptr ;

/**
SGeo::SetLastUploadCFBase
---------------------------

Canonically invoked from CSGFoundry::upload with CSGFoundry::getOriginCFBase

**/

void SGeo::SetLastUploadCFBase(const char* cfbase)
{
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






