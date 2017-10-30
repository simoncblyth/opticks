#include "GLMFormat.hpp"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"

#include "NEmitConfig.hpp"
#include "NEmitPhotonsNPY.hpp"

#include "PLOG.hh"


NEmitPhotonsNPY::NEmitPhotonsNPY(NCSG* csg)
    :
    m_csg(csg),
    m_emit(csg->emit()),
    m_cfg_( csg->emitconfig() ),
    m_cfg( new NEmitConfig( m_cfg_ )),
    m_root( csg->getRoot()),
    m_data(NPY<float>::make(m_cfg->photons, 4, 4))
{
    init();
}

NPY<float>* NEmitPhotonsNPY::getNPY() const 
{
    return m_data ; 
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

    m_data->zero();   

    m_cfg->dump();

    unsigned numPhoton = m_data->getNumItems();
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

    for(unsigned i=0 ; i < numPhoton ; i++)
    {   
        const glm::vec3& pos = points[i] ; 
        const glm::vec3& nrm = normals[i] ; 

        glm::vec3 dir(nrm) ; 
        dir *= fdir ; 

        glm::vec3 adir(dir);
        glm::vec3 least_parallel_axis(0) ; 

        if( adir.x <= adir.y && adir.x <= adir.z )
        {
            least_parallel_axis.x = 1.f ; 
        }
        else if( adir.y <= adir.x && adir.y <= adir.z )
        {
            least_parallel_axis.y = 1.f ; 
        }
        else
        {
            least_parallel_axis.z = 1.f ; 
        }

        glm::vec3 pol = glm::normalize( glm::cross( least_parallel_axis, dir )) ; 

        if(i<10)
        {
            std::cout << " i " << std::setw(6) << i 
                      << " pos " << gpresent(pos)
                      << " nrm " << gpresent(nrm)
                      << " dir " << gpresent(dir)
                      << " adir " << gpresent(adir)
                      << " lpa " << gpresent(least_parallel_axis)
                      << " pol " << gpresent(pol)
                      << std::endl 
                      ;

        }

    

        glm::vec4 q0(     pos.x,      pos.y,      pos.z,  ftime );
        glm::vec4 q1(     dir.x,      dir.y,      dir.z,  fweight );
        glm::vec4 q2(     pol.x,      pol.y,      pol.z,  fwavelength );
        glm::uvec4 u3(   0,0,0,0 );   // flags 

        m_data->setQuad( q0, i, 0 );
        m_data->setQuad( q1, i, 1 );
        m_data->setQuad( q2, i, 2 );
        m_data->setQuad( u3, i, 3 );  
    }   
}


NPY<float>* NEmitPhotonsNPY::make(NCSG* csg)
{
    NEmitPhotonsNPY* ep = new NEmitPhotonsNPY(csg);
    return ep->getNPY();
}



