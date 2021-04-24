#include "Opticks.hh"
#include "Composition.hh"
#include "SRenderer.hh"
#include "SMeta.hh"
#include "NP.hh"
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
    m_config(config),
    m_outdir(m_ok->getSnapOutDir()),
    m_nameprefix(m_ok->getNamePrefix()),
    m_meta(new SMeta)
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
        save(); 
    }
    
    LOG(info) << "]" ;
}


void Snap::render_one(const char* path)
{
    double dt = m_renderer->render();   
    m_frame_times.push_back(dt); 

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

void Snap::getMinMaxAvg(double& mn, double& mx, double& av) const 
{
    const std::vector<double>& t = m_frame_times ; 

    typedef std::vector<double>::const_iterator IT ;     
    IT mn_ = std::min_element( t.begin(), t.end()  ); 
    IT mx_ = std::max_element( t.begin(), t.end()  ); 
    double sum = std::accumulate(t.begin(), t.end(), 0. );   

    mn = *mn_ ; 
    mx = *mx_ ; 
    av = t.size() > 0 ? sum/double(t.size()) : -1. ;  
}

void Snap::save() const 
{
    nlohmann::json& js = m_meta->js ; 

    js["argline"] = m_ok->getArgLine(); 
    js["cfg"] = m_config->getCfg(); 
    js["nameprefix"] = m_nameprefix  ;  
    js["emm"] = m_ok->getEnabledMergedMesh() ;  

    double mn, mx, av ; 
    getMinMaxAvg(mn, mx, av); 

    js["mn"] = mn ;  
    js["mx"] = mx ;  
    js["av"] = av ;  

    std::string js_name = m_nameprefix ; 
    js_name += ".json" ; 
    m_meta->save(m_outdir,  js_name.c_str() ); 
        
    std::string np_name = m_nameprefix ; 
    np_name += ".npy" ;  

    NP::Write(m_outdir, np_name.c_str(), (double*)m_frame_times.data(),  m_frame_times.size() );  
}



