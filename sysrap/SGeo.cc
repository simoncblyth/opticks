#include <cstring>
#include "SPath.hh"
#include "SProc.hh"
#include "SGeo.hh"

const char* SGeo::LAST_UPLOAD_CFBASE = nullptr ;

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
    const char* execname = SProc::ExecutableName(); 
    const char* outdir = SPath::Resolve(cfbase, execname, DIRPATH );  
    return outdir ; 
}






