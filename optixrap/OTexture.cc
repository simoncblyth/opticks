#include "OKConf.hh"
#include "PLOG.hh"
#include "NPY.hpp"
#include "OFormat.hh"
#include "OTexture.hh"

const plog::Severity OTexture::LEVEL = PLOG::EnvLevel("OTexture", "DEBUG") ; 

template <typename T>
void OTexture::Upload2DLayeredTexture(optix::Context& context, const char* param_key, const char* domain_key, const NPYBase* inp)
{
    unsigned nd = inp->getDimensions(); 
    assert( nd == 4 );    

    const unsigned ni = inp->getShape(0);  // number of texture layers
    const unsigned nj = inp->getShape(1);  // width 
    const unsigned nk = inp->getShape(2);  // height
    const unsigned nl = inp->getShape(3);  // components
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

    unsigned bufferdesc = RT_BUFFER_INPUT | RT_BUFFER_LAYERED ; 
    //  If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
    optix::Buffer texBuffer = context->createBuffer(bufferdesc); 

    RTformat format = OFormat::TextureFormat<T>(nl);
    texBuffer->setFormat( format ); 
    texBuffer->setSize(nj, nk, ni);      // 3rd depth arg is number of layers

    // attempt at using mapEx failed, so upload all layers at once 
    void* tex_data = texBuffer->map() ; 
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

    RTfiltermode minmag = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
    RTfiltermode minification = minmag ; 
    RTfiltermode magnification = minmag ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ; 

    tex->setFilteringModes(minification, magnification, mipmapping);

    // indexmode : controls the interpretation of texture coordinates
    //RTtextureindexmode indexmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ;  // parametrized over [0,1]
    RTtextureindexmode indexmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // texture coordinates are interpreted as array indices into the contents of the underlying buffer object
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


template OXRAP_API void OTexture::Upload2DLayeredTexture<unsigned char>(optix::Context&, const char*, const char*, const NPYBase* );

