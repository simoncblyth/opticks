#include "NumpyEvt.hpp"

#include "uif.h"
#include "NPY.hpp"
#include "VecNPY.hpp"
#include "MultiVecNPY.hpp"

#include <sstream>


void NumpyEvt::setGenstepData(NPY* genstep)
{
    m_genstep_data = genstep  ;
    m_genstep_attr = new MultiVecNPY();
    m_genstep_attr->add(new VecNPY("vpos",m_genstep_data,1,0));    // (x0, t0)                     2nd GenStep quad 
    m_genstep_attr->add(new VecNPY("vdir",m_genstep_data,2,0));    // (DeltaPosition, step_length) 3rd GenStep quad

    m_num_photons = m_genstep_data->getUSum(0,3);

    NPY* npy = NPY::make_vec4(m_num_photons);

    setPhotonData(npy);   

    // stuff genstep index into the photon allocation 
    // to allow generation to access appropriate genstep 

    unsigned int srclen = m_genstep_data->getShape(0);
    unsigned int dstlen = m_photon_data->getShape(0);
    printf("NumpyEvt::setGenstepData srclen %u (steps) dstlen %u (photons) \n", srclen, dstlen ); 

    // repeat step index for every photon
    unsigned int photon(0) ;
    for(unsigned int step=0 ; step < srclen ; step++)
    {
        unsigned int npho = m_genstep_data->getUInt(step, 0, 3);
        assert(npho > 0 && npho < 150); // by observation of Cerenkov steps
        for(unsigned int n=0 ; n < npho ; ++n)
        { 
            assert(photon < dstlen);
            m_photon_data->setUInt(photon, 0,0, step );
            photon += 1 ;         
        }
    }
    printf(" photon %u m_num_photons %u \n", photon, m_num_photons );
    assert(photon == m_num_photons ); 
    // not m_num_photons-1 as last incremented photon value not used by setUInt
}

void NumpyEvt::setPhotonData(NPY* photon_data)
{
    m_photon_data = photon_data  ;
    m_photon_attr = new MultiVecNPY();
    m_photon_attr->add(new VecNPY("vpos",m_photon_data,0,0));
}


void NumpyEvt::dumpPhotonData()
{
    if(!m_photon_data) return ;
    dumpPhotonData(m_photon_data);
}

void NumpyEvt::dumpPhotonData(NPY* photons)
{
    std::cout << photons->description("NumpyEvt::dumpPhotonData") << std::endl ;

    for(unsigned int i=0 ; i < photons->getShape(0) ; i++)
    {
        if(i%10000 == 0)
        {
            unsigned int ux = photons->getUInt(i,0,0); 
            float fx = photons->getFloat(i,0,0); 
            float fy = photons->getFloat(i,0,1); 
            float fz = photons->getFloat(i,0,2); 
            float fw = photons->getFloat(i,0,3); 
            printf(" ph  %7u   ux %7u   fxyzw %10.3f %10.3f %10.3f %10.3f \n", i, ux, fx, fy, fz, fw );             
        }
    }  
}



std::string NumpyEvt::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " ;
    if(m_genstep_data)  ss << m_genstep_data->description("m_genstep_data") ;
    if(m_photon_data)   ss << m_photon_data->description("m_photon_data") ;
    return ss.str();
}



