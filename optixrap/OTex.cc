#include "OKConf.hh"
#include "SStr.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "OFormat.hh"
#include "OTex.hh"

#include "OCtx.hh"


const plog::Severity OTex::LEVEL = PLOG::EnvLevel("OTex", "DEBUG") ; 


/**
OTex::Upload2DLayeredTexture
---------------------------------

Note reversed shape order of the texBuffer->setSize( width, height, depth)
wrt to the shape of the input buffer.  

For example with a landscape input PPM image of height 512 and width 1024 
the natural array shape to use is (height, width, ncomp) ie (512,1024,3) 
This is natural because it matches the row-major ordering of the image data 
in PPM files starting with the top row (with a width) and rastering down 
*height* by rows. 

BUT when specifying the dimensions of the tex buffer it is necessary to use::

     texBuffer->setSize(width, height, depth) 

NB do not like having optix::Context in the interface..  it is OK to 
use that within the implementation but need to avoid having it 
in the interface... but without it in the interface cannot return it 
from the opaque type : so need to wrap absolutely everything ?

* Can cheat with void* of course ?

**/


void OTex::Upload2DLayeredTexture(const char* param_key, const char* domain_key, const NPYBase* inp, const char* config)
{
    float xmin = inp->getMeta<float>("xmin", "0.") ; 
    float xmax = inp->getMeta<float>("xmax", "1.") ; 
    float ymin = inp->getMeta<float>("ymin", "0.") ; 
    float ymax = inp->getMeta<float>("ymax", "1.") ; 
    LOG(info) << " xmin " << xmin << " xmax " << xmax << " ymin " << ymin << " ymax " << ymax ;
    OCtx_set_float4(domain_key, xmin, xmax, ymin, ymax);  

    void* buffer_ptr = OCtx_create_buffer(inp, NULL, 'I', 'L' ); 
    unsigned tex_id = OCtx_create_texture_sampler(buffer_ptr, config ); 
    OCtx_set_texture_param( buffer_ptr, tex_id, param_key );  
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



