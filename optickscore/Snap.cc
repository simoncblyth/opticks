#include "Opticks.hh"
#include "Composition.hh"
#include "SRenderer.hh"
#include "NSnapConfig.hpp"

#include "PLOG.hh"
#include "Snap.hh"

const plog::Severity Snap::LEVEL = PLOG::EnvLevel("Snap", "DEBUG"); 


Snap::Snap(Opticks* ok, SRenderer* renderer, NSnapConfig* config)  
    : 
    m_ok(ok), 
    m_composition(m_ok->getComposition()), 
    m_numsteps(m_ok->getSnapSteps()),
    m_renderer(renderer), 
    m_config(config)
{
}
 
void Snap::render()
{
    LOG(info)  << "[" << m_config->desc() ; 

    if( m_numsteps == 0)
    {   
        const char* path = m_ok->getSnapPath(0);
        render_one(path);
    }
    else
    {   
        render_many();
    }
    
    //m_otracer->report("OpTracer::snap");   // saves for runresultsdir
    
    if(!m_ok->isProduction())
    {   
        m_ok->saveParameters();
    }
    
    LOG(info) << "]" ;
}


void Snap::render_one(const char* path)
{
    double dt = m_renderer->render();   

    std::string top_annotation = m_ok->getContextAnnotation();
    std::string bottom_annotation = m_ok->getFrameAnnotation(0, 1, dt );
    unsigned anno_line_height = m_ok->getAnnoLineHeight() ;

    LOG(info) << top_annotation << " | " << bottom_annotation << " | " << path << " | " << anno_line_height ;

    m_renderer->snap(path, bottom_annotation.c_str(), top_annotation.c_str(), anno_line_height );
}

void Snap::render_many()
{
    m_eyes.clear(); 
    m_ok->getSnapEyes(m_eyes); 
    render_many(m_eyes);           
}

void Snap::render_many(const std::vector<glm::vec3>& eyes )
{
    char path[128] ; 
    const char* path_fmt = m_ok->getSnapPath(-1);

    for(int i=0 ; i < int(eyes.size()) ; i++)
    {
        const glm::vec3& eye = eyes[i] ;
        m_composition->setEye( eye.x, eye.y, eye.z );

        snprintf(path, 128, path_fmt, i );
        render_one(path);
    }
}


