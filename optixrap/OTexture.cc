#include "OKConf.hh"
#include "SStr.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "OFormat.hh"
#include "OTex.hh"
#include "OTexture.hh"

const plog::Severity OTexture::LEVEL = PLOG::EnvLevel("OTexture", "DEBUG") ; 

/**
OTexture::Upload2DLayeredTexture
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
**/


template <typename T>
void OTexture::Upload2DLayeredTexture(optix::Context& context, const char* param_key, const char* domain_key, const NPYBase* inp, const char* config)
{
    unsigned nd = inp->getDimensions(); 
    assert( nd == 4 );    

    const unsigned ni = inp->getShape(0);  // number of texture layers
    const unsigned nj = inp->getShape(1);  // height 
    const unsigned nk = inp->getShape(2);  // width
    const unsigned nl = inp->getShape(3);  // components

    const unsigned depth  = ni ; 
    const unsigned height = nj ; 
    const unsigned width  = nk ; 
    const unsigned ncomp  = nl ; 

    LOG(info) 
        << " inp " << inp->getShapeString()
        << " depth:ni  " << depth 
        << " height:nj " << height
        << " width:nk " << width 
        << " ncomp:nl " << ncomp 
        ;

    assert( nl < 5 );   

    float xmin = inp->getMeta<float>("xmin", "0.") ; 
    float xmax = inp->getMeta<float>("xmax", "1.") ; 
    float ymin = inp->getMeta<float>("ymin", "0.") ; 
    float ymax = inp->getMeta<float>("ymax", "1.") ; 

    context[domain_key]->setFloat(optix::make_float4(xmin, xmax, ymin, ymax));

    LOG(info) 
        << " xmin " << xmin 
        << " xmax " << xmax 
        << " ymin " << ymin 
        << " ymax " << ymax 
        ;


    bool layered = true ; 
    unsigned bufferdesc = RT_BUFFER_INPUT ; 
    if(layered) bufferdesc |= RT_BUFFER_LAYERED ; 
    optix::Buffer texBuffer = context->createBuffer(bufferdesc); 
    RTformat format = OFormat::Get<T>(nl);
    texBuffer->setFormat( format ); 

    if(layered){ 
        LOG(info) << " layered texBuffer->setSize(width:nk, height:nj, depth:ni) " << "(" << width << " " << height << " " << depth << ")" ; 
        texBuffer->setSize(width, height, depth);  // when layered, 3rd depth arg is number of layers
    } else {
        LOG(info) << " non-layered setSize(width:nk, height:nj) " << "(" << width << " " << height << ")" ; 
        texBuffer->setSize(width, height);   
    }
    void* tex_data = texBuffer->map() ; // attempt at using mapEx failed, so upload all layers at once 
    inp->write_(tex_data); 
    texBuffer->unmap(); 

    LOG(LEVEL) << "[ creating tex_sampler " ;  
    optix::TextureSampler tex = context->createTextureSampler(); 

    //RTwrapmode wrapmode = RT_WRAP_REPEAT ; 
    RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ; 
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ; 
    tex->setWrapMode(0, wrapmode);
    tex->setWrapMode(1, wrapmode);
    //tex->setWrapMode(2, wrapmode);   corresponds to layer?

    RTfiltermode filtermode = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
    RTfiltermode minification = filtermode ; 
    RTfiltermode magnification = filtermode ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ; 

    tex->setFilteringModes(minification, magnification, mipmapping);

    RTtextureindexmode indexmode = (RTtextureindexmode)OTex::IndexMode(config) ;  
    LOG(info) << "tex.setIndexingMode [" << OTex::IndexModeString(indexmode) << "]" ; 
    tex->setIndexingMode( indexmode );  


    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ; // return floating point values normalized by the range of the underlying type
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;  // return data of the type of the underlying buffer
    // when the underlying type is float the is no difference between RT_TEXTURE_READ_NORMALIZED_FLOAT and RT_TEXTURE_READ_ELEMENT_TYPE

    tex->setReadMode( readmode ); 
    tex->setMaxAnisotropy(1.0f);
    LOG(LEVEL) << "] creating tex_sampler " ;  

    unsigned deprecated0 = 0 ; 
    unsigned deprecated1 = 0 ; 
    tex->setBuffer(deprecated0, deprecated1, texBuffer); 

    unsigned tex_id = tex->getId() ; 

    optix::int4 param = optix::make_int4(ni, nj, nk, tex_id); 
    context[param_key]->setInt(param);

    LOG(info) 
        << param_key 
        << " ( " 
        << param.x << " "
        << param.y << " "
        << param.z << " "
        << param.w << " "
        << " ) "
        << " ni/nj/nk/tex_id "
        ; 
}


template OXRAP_API void OTexture::Upload2DLayeredTexture<unsigned char>(optix::Context&, const char*, const char*, const NPYBase*, const char* );

