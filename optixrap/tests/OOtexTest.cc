#include "NPY.hpp"
#include "OXPPNS.hh"
#include "OConfig.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);    

    LOG(info) << " ok " ; 


    optix::Context context = optix::Context::create();

    int nx = 16 ; 
    int ny = 16 ; 
    int nz = 1 ; 

    optix::Buffer texBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx, ny);
    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, nx, ny);

    float* data = (float *) texBuffer->map();

    for(int i=0 ; i < nx ; i++){
    for(int j=0 ; j < ny ; j++){
    for(int k=0 ; k < nz ; k++)
    {
       int index = i*ny*nz + j*nz + k ;
       *(data + index) = float(index) ;
    }
    }
    }

    texBuffer->unmap(); 


    optix::TextureSampler tex = context->createTextureSampler();

    tex->setWrapMode(0, RT_WRAP_REPEAT);
    tex->setWrapMode(1, RT_WRAP_REPEAT);
    tex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

    //RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_ARRAY_INDEX ; 
    RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ; 
    tex->setIndexingMode(indexingmode);  

    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ;
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;    // No conversion
    tex->setReadMode(readmode);

    tex->setMaxAnisotropy(1.0f);
    tex->setMipLevelCount(1);
    tex->setArraySize(1);
    tex->setBuffer(0, 0, texBuffer);

    context["some_texture"]->setTextureSampler(tex);
    context["out_buffer"]->setBuffer(outBuffer);   

    int tex_id = tex->getId();
    context["tex_param"]->setInt(optix::make_int4(tex_id, 0, 0, 0 ));


    OConfig* cfg = new OConfig(context); 

    bool defer = true ;
    unsigned int entry = cfg->addEntry("texTest.cu.ptx", "texTest", "exception", defer);

    context->setEntryPointCount(1) ;

    cfg->dump();
    cfg->apply();     


    context->setPrintEnabled(true);
    context->setPrintBufferSize(8192);

    context->validate();
    context->compile();
    context->launch(entry,  0, 0);
    context->launch(entry, nx, ny);


    NPY<float>* npy = NPY<float>::make(nx, ny);
    npy->read( outBuffer->map() );
    outBuffer->unmap(); 

    npy->dump();
    npy->save("$TMP/texTest.npy");

    LOG(info) << "DONE" ; 

    return 0 ;     
}

