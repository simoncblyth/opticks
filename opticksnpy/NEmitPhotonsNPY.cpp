#include "GLMFormat.hpp"

#include "NGLMExt.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"


#include "FabStepNPY.hpp"
#include "NEmitConfig.hpp"
#include "NEmitPhotonsNPY.hpp"

#include "PLOG.hh"


NEmitPhotonsNPY::NEmitPhotonsNPY(NCSG* csg, unsigned gencode, bool emitdbg)
    :
    m_csg(csg),
    m_emitdbg(emitdbg),
    m_emit(csg->emit()),
    m_cfg_( csg->emitconfig() ),
    m_cfg( new NEmitConfig( m_cfg_ )),
    m_root( csg->getRoot()),
    m_photons(NPY<float>::make(m_cfg->photons, 4, 4)),
    m_fabstep(new FabStepNPY(gencode, 1, m_cfg->photons)),
    m_fabstep_npy(m_fabstep->getNPY())
{
    init();
}

NPY<float>* NEmitPhotonsNPY::getPhotons() const 
{
    return m_photons ; 
}
FabStepNPY* NEmitPhotonsNPY::getFabStep() const 
{
    return m_fabstep ; 
}
NPY<float>* NEmitPhotonsNPY::getFabStepData() const 
{
    return m_fabstep_npy ; 
}



std::string NEmitPhotonsNPY::desc() const 
{
    std::stringstream ss ;
    ss << m_cfg->desc() ; 
    return ss.str();
}


void NEmitPhotonsNPY::init()
{
    assert( m_emit == 1 || m_emit == -1 );

    m_photons->zero();   

    if(m_emitdbg)
    m_cfg->dump();

    unsigned numPhoton = m_photons->getNumItems();

    LOG(info) << desc() 
              << " numPhoton " << numPhoton 
               ;

    std::vector<glm::vec3> points ; 
    std::vector<glm::vec3> normals ; 
    m_root->generateParPoints( points, normals, numPhoton );

    assert( points.size() == numPhoton );
    assert( normals.size() == numPhoton );

    float fdir = float(m_emit);  // +1 out -1 in 
    float ftime = m_cfg->time ;  // ns
    float fweight = m_cfg->weight ;
    float fwavelength = m_cfg->wavelength ; // nm
    float fposdelta = m_cfg->posdelta ; 

    for(unsigned i=0 ; i < numPhoton ; i++)
    {   
        const glm::vec3& nrm = normals[i] ; 

        glm::vec3 pos(points[i]);

        glm::vec3 dir(nrm) ; 
        dir *= fdir ; 

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




