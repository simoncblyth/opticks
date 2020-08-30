#include "OKConf.hh"
#include "SStr.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "OFormat.hh"
#include "OTex.hh"

#include "OCtx.hh"


const plog::Severity OTex::LEVEL = PLOG::EnvLevel("OTex", "INFO") ; 


void OTex::UploadDomainFloat4(const char* domain_key, const NPYBase* inp)
{
    float xmin = inp->getMeta<float>("xmin", "0.") ; 
    float xmax = inp->getMeta<float>("xmax", "1.") ; 
    float ymin = inp->getMeta<float>("ymin", "0.") ; 
    float ymax = inp->getMeta<float>("ymax", "1.") ; 
    LOG(info) << " xmin " << xmin << " xmax " << xmax << " ymin " << ymin << " ymax " << ymax ;

    OCtx octx ; 
    octx.set_context_float4(domain_key, xmin, xmax, ymin, ymax);  
}




/**
OTex::IndexMode
---------------------

indexmode : controls the interpretation of texture coordinates

**/

int OTex::IndexMode( const char* config )
{
    RTtextureindexmode indexmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ;  
    if(SStr::Contains(config, INDEX_NORMALIZED_COORDINATES )) 
    {
        indexmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ; // parametrized over [0,1] 
    }
    else if(SStr::Contains(config, INDEX_ARRAY_INDEX))
    {
        indexmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // array indices into the contents
    }
    return (int)indexmode ; 
}

const char* OTex::INDEX_NORMALIZED_COORDINATES = "INDEX_NORMALIZED_COORDINATES" ; 
const char* OTex::INDEX_ARRAY_INDEX            = "INDEX_ARRAY_INDEX" ; 
const char* OTex::IndexModeString( int indexmode_ )
{
    const char* s = NULL ; 
    RTtextureindexmode indexmode = (RTtextureindexmode)indexmode_ ; 
    switch(indexmode)
    {
       case RT_TEXTURE_INDEX_NORMALIZED_COORDINATES: s = INDEX_NORMALIZED_COORDINATES ; break ; 
       case RT_TEXTURE_INDEX_ARRAY_INDEX:            s = INDEX_ARRAY_INDEX            ; break ; 
    } 
    return s ; 
}



