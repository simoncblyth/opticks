#include "NumpyEvt.hpp"

#include "NPY.hpp"
#include "VecNPY.hpp"
#include "MultiVecNPY.hpp"

#include <sstream>


void NumpyEvt::setGenstepData(NPY* genstep_data)
{
    m_genstep_data = genstep_data  ;
    m_genstep_attr = new MultiVecNPY();
    m_genstep_attr->add(new VecNPY("vpos",m_genstep_data,1,0));    // (x0, t0)                     2nd GenStep quad 
    m_genstep_attr->add(new VecNPY("vdir",m_genstep_data,2,0));    // (DeltaPosition, step_length) 3rd GenStep quad

    m_num_photons = m_genstep_data->getUSum(0,3);
    allocatePhotonData();
}

void NumpyEvt::setPhotonData(NPY* photon_data)
{
    m_photon_data = photon_data  ;
    m_photon_attr = new MultiVecNPY();
    m_photon_attr->add(new VecNPY("vpos",m_photon_data,0,0));
}

void NumpyEvt::allocatePhotonData()
{
    NPY* npy = NPY::make_vec4(m_num_photons);
    //setPhotonData(npy);   // hmm segmenting

    // TODO: uif_t stuff genstep index into the photon allocation 
    //       to allow generation to access appropriate genstep 
    //
}


std::string NumpyEvt::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " ;
    if(m_genstep_data)  ss << m_genstep_data->description("m_genstep_data") ;
    if(m_photon_data)   ss << m_photon_data->description("m_photon_data") ;
    return ss.str();
}



