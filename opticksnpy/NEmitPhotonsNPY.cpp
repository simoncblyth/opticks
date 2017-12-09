
#include "BHex.hh"
#include "GLMFormat.hpp"

#include "NGLMExt.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NRngDiffuse.hpp"


#include "FabStepNPY.hpp"
#include "NEmitConfig.hpp"
#include "NEmitPhotonsNPY.hpp"

#include "PLOG.hh"


NEmitPhotonsNPY::NEmitPhotonsNPY(NCSG* csg, unsigned gencode, unsigned seed, bool emitdbg, NPY<unsigned>* mask)
    :
    m_csg(csg),
    m_gencode(gencode),
    m_seed(seed),
    m_emitdbg(emitdbg),
    m_mask(mask),
    m_emit( csg->getEmit() ),
    m_emitcfg_( csg->getEmitConfig() ),
    m_emitcfg( new NEmitConfig( m_emitcfg_ )),
    m_root( csg->getRoot()),
    m_photons(NPY<float>::make(m_emitcfg->photons, 4, 4)),
    m_photons_masked(NULL),
    m_fabstep(new FabStepNPY(gencode, 1, m_emitcfg->photons)),  // code, num_step, num_photons_per_step
    m_fabstep_masked(NULL),
    m_diffuse( m_emitcfg->diffuse ? new NRngDiffuse(m_seed+100,m_emitcfg->ctmindiffuse, m_emitcfg->ctmaxdiffuse) : NULL )  
{
    init();
    
    if(m_mask)
    {
        m_photons_masked = NPY<float>::make_masked( m_photons, m_mask ) ; 
        m_fabstep_masked = new FabStepNPY(m_gencode, 1, m_mask->getShape(0)) ;
    } 
}


NPY<float>* NEmitPhotonsNPY::getPhotons() const 
{
    return m_mask ? m_photons_masked : m_photons ; 
}  
NPY<float>* NEmitPhotonsNPY::getPhotonsRaw() const 
{
    return m_photons ; 
}


FabStepNPY* NEmitPhotonsNPY::getFabStep() const 
{
    return m_mask ? m_fabstep_masked : m_fabstep  ; 
}
FabStepNPY* NEmitPhotonsNPY::getFabStepRaw() const 
{
    return m_fabstep  ; 
}


NPY<float>* NEmitPhotonsNPY::getFabStepData() const 
{
    FabStepNPY* fs = getFabStep() ;
    return fs->getNPY() ; 
}

NPY<float>* NEmitPhotonsNPY::getFabStepRawData() const 
{
    FabStepNPY* fs = getFabStepRaw() ;
    return fs->getNPY() ; 
}






std::string NEmitPhotonsNPY::desc() const 
{
    std::stringstream ss ;
    ss << m_emitcfg->desc() ; 
    return ss.str();
}


void NEmitPhotonsNPY::init()
{
    assert( m_emit == 1 || m_emit == -1 );

    m_photons->zero();   

    if(m_emitdbg) m_emitcfg->dump();

    unsigned numPhoton = m_photons->getNumItems();

    LOG(debug) << desc() 
              << " numPhoton " << numPhoton 
               ;

    std::vector<glm::vec3> points ; 
    std::vector<glm::vec3> normals ; 

    std::string sheetmask_ = m_emitcfg->sheetmask ; 
    unsigned sheetmask = BHex<unsigned>::hex_lexical_cast( sheetmask_.c_str() ) ;
    glm::vec4 uvdom( m_emitcfg->umin, m_emitcfg->umax, m_emitcfg->vmin, m_emitcfg->vmax );

    m_root->generateParPoints( m_seed, uvdom, points, normals, numPhoton, sheetmask );

    assert( points.size() == numPhoton );
    assert( normals.size() == numPhoton );

    float fdir = float(m_emit);  // +1 out -1 in 
    float ftime = m_emitcfg->time ;  // ns
    float fweight = m_emitcfg->weight ;
    float fwavelength = m_emitcfg->wavelength ; // nm
    float fposdelta = m_emitcfg->posdelta ; 

    for(unsigned i=0 ; i < numPhoton ; i++)
    {   
        const glm::vec3& nrm = normals[i] ; 

        glm::vec3 pos(points[i]);

        glm::vec3 dir(nrm) ; 
        dir *= fdir ; 

        if(m_diffuse)
        {  
            glm::vec4 u ; 
            int trials(0) ; 
            float udotd = m_diffuse->diffuse(u, trials, dir); 
            dir.x = u.x ; 
            dir.y = u.y ; 
            dir.z = u.z ; 
            if( i < 10 ) std::cout << " diffuse udotd " << udotd << std::endl ; 
        } 

        if(fposdelta != 0.)  // nudge photon start position along its direction 
        {
            pos += dir*fposdelta ; 
        }

        //bool dump = i < 10 ; 
        bool dump = false ; 
        glm::vec3 pol = nglmext::pick_transverse_direction( dir, dump );

/*
        if(i<10)
        {
            std::cout << " i " << std::setw(6) << i 
                      << " pos " << gpresent(pos)
                      << " nrm " << gpresent(nrm)
                      << " dir " << gpresent(dir)
                      << " pol " << gpresent(pol)
                      << std::endl 
                      ;
        }
*/  

        glm::vec4 q0(     pos.x,      pos.y,      pos.z,  ftime );
        glm::vec4 q1(     dir.x,      dir.y,      dir.z,  fweight );
        glm::vec4 q2(     pol.x,      pol.y,      pol.z,  fwavelength );
        glm::uvec4 u3(   0,0,0,0 );   // flags 

        m_photons->setQuad( q0, i, 0 );
        m_photons->setQuad( q1, i, 1 );
        m_photons->setQuad( q2, i, 2 );
        m_photons->setQuad( u3, i, 3 );  
    }   
}




