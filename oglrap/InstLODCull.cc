#include <sstream>



#include <GL/glew.h>



#include "NPY.hpp"
#include "NSlice.hpp"

#include "GDrawable.hh"
#include "GMergedMesh.hh"

#include "InstLODCull.hh"

#include "PLOG.hh"

const unsigned InstLODCull::INSTANCE_MINIMUM = 10000 ; 

const unsigned InstLODCull::LOC_InstanceTransform = 0 ;


InstLODCull::InstLODCull(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),
    m_composition(NULL),
    m_drawable(NULL),
    m_geometry(NULL),
    m_itransforms(NULL),
    m_num_instance(0),
    m_enabled(false),
    m_fork(NULL)
{
}


InstLODCull::~InstLODCull()
{
}

void InstLODCull::setComposition(Composition* composition)
{
    m_composition = composition ;
}
Composition* InstLODCull::getComposition()
{
    return m_composition ;
}


void InstLODCull::setITransformsBuffer(NPY<float>* ibuf)
{
    assert(ibuf);
    m_itransforms = ibuf ; 
    m_num_instance = ibuf->getNumItems();
    m_enabled = m_num_instance > INSTANCE_MINIMUM  ;
}

bool InstLODCull::isEnabled() const 
{
    return m_enabled ; 
}

std::string InstLODCull::desc() const 
{
    std::stringstream ss ; 
    ss << " InstLODCull"
       << ( m_enabled ? " ENABLED " : " DISABLED " )
       << " num_instance " << m_num_instance
       << " INSTANCE_MINIMUM " << INSTANCE_MINIMUM
       ;

    return ss.str();
}


void InstLODCull::upload(GMergedMesh* mm, bool /*debug*/)
{
    m_geometry = mm ;
    m_drawable = static_cast<GDrawable*>(mm);

    NPY<float>* ibuf = mm->getITransformsBuffer();
    setITransformsBuffer(ibuf);

    if(m_verbosity > 0)
    LOG(info) << "InstLODCull::upload"
              << desc()
              ;

    if(!isEnabled()) return ; 

    NSlice* islice = m_geometry->getInstanceSlice();
    NSlice* fslice = m_geometry->getFaceSlice();
    assert(islice == NULL);
    assert(fslice == NULL);

    initShader();
    
    //upload_buffers(islice, fslice);

}



void InstLODCull::initShader()
{
    if(m_verbosity > 0)
    LOG(info) << "InstLODCull::initShader" ;

    create_shader();
    setNoFrag(true);


    const char *vars[] = {
                           "VizTransform0LOD0",
                           "VizTransform1LOD0",
                           "VizTransform2LOD0",
                           "VizTransform3LOD0",
                           "gl_NextBuffer",
                           "VizTransform0LOD1",
                           "VizTransform1LOD1",
                           "VizTransform2LOD1",
                           "VizTransform3LOD1",
                           "gl_NextBuffer",
                           "VizTransform0LOD2",
                           "VizTransform1LOD2",
                           "VizTransform2LOD2",
                           "VizTransform3LOD2"
                         };

    glTransformFeedbackVaryings(m_program, 14, vars, GL_INTERLEAVED_ATTRIBS);

    glBindAttribLocation(m_program, LOC_InstanceTransform, "InstanceTransform");

    link_shader();

} 


// /Users/blyth/env/graphics/opengl/instcull/LODCullShader.cc


