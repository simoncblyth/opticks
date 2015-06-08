#include "NumpyEvt.hpp"

#include "uif.h"
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "VecNPY.hpp"
#include "MultiVecNPY.hpp"

#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void NumpyEvt::setGenstepData(NPY<float>* genstep)
{
    G4StepNPY gs(genstep);  

    m_genstep_data = genstep  ;
    m_genstep_attr = new MultiVecNPY();
    m_genstep_attr->add(new VecNPY("vpos",m_genstep_data,1,0));    // (x0, t0)                     2nd GenStep quad 
    m_genstep_attr->add(new VecNPY("vdir",m_genstep_data,2,0));    // (DeltaPosition, step_length) 3rd GenStep quad

    m_num_photons = m_genstep_data->getUSum(0,3);

    NPY<float>* pho = NPY<float>::make_vec4(m_num_photons, 4); // must match GPU side photon.h:PQUAD
    setPhotonData(pho);   

    NPY<short>* rec = NPY<short>::make_vec4(m_num_photons, 2); 
    setRecordData(rec);   


    // stuff genstep index into the photon allocation 
    // to allow generation to access appropriate genstep 

    unsigned int numStep   = m_genstep_data->getShape(0);
    unsigned int numPhoton = m_photon_data->getShape(0);

    unsigned int count(0) ;
    for(unsigned int index=0 ; index < numStep ; index++)
    {
        unsigned int npho = m_genstep_data->getUInt(index, 0, 3);
        if(gs.isCerenkovStep(index))
        {
            assert(npho > 0 && npho < 150);      // by observation of Cerenkov steps
        }
        else if(gs.isScintillationStep(index))
        {
            assert(npho >= 0 && npho < 1000);     // by observation of Scintillation steps                  
        } 

        for(unsigned int n=0 ; n < npho ; ++n)
        { 
            assert(count < numPhoton);
            m_photon_data->setUInt(count, 0,0, index );  // set "phead" : repeating step index for every photon to be generated for the step
            count += 1 ;         
        }  // over photons for each step
    }      // over gen steps


    LOG(info) << "NumpyEvt::setGenstepData " 
              << " stepId(0) " << gs.getStepId(0) 
              << " genstep length " << numStep 
              << " photon length " << numPhoton
              << "  num_photons " << m_num_photons  ; 

    assert(count == m_num_photons ); 
    assert(count == numPhoton ); 
    // not m_num_photons-1 as last incremented count value is not used by setUInt
}

void NumpyEvt::setPhotonData(NPY<float>* photon_data)
{
    m_photon_data = photon_data  ;
    m_photon_attr = new MultiVecNPY();
    unsigned int size = 4 ; 
    m_photon_attr->add(new VecNPY("vpos",m_photon_data,0,0,size));      // 1st quad
    m_photon_attr->add(new VecNPY("vdir",m_photon_data,1,0,size));      // 2nd quad
    m_photon_attr->add(new VecNPY("vpol",m_photon_data,2,0,size));      // 3rd quad
    m_photon_attr->add(new VecNPY("iflg",m_photon_data,3,0,size,'i'));  // 4th quad

    // corresponds to GPU side cu/photon.h:psave 
}

void NumpyEvt::setRecordData(NPY<short>* record_data)
{
    m_record_data = record_data  ;
    m_record_attr = new MultiVecNPY();
    unsigned int size = 4 ; 
    m_record_attr->add(new VecNPY("rpos",m_record_data,0,0,size,'s',true));    // 4*signed short int to be normalized into -1.f:1.f by OpenGL     (1st half-quad)
    m_record_attr->add(new VecNPY("rflg",m_record_data,1,0,size,'s',false));   // 4*signed short int                                              (2nd half-quad)
}






void NumpyEvt::dumpPhotonData()
{
    if(!m_photon_data) return ;
    dumpPhotonData(m_photon_data);
}

void NumpyEvt::dumpPhotonData(NPY<float>* photons)
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



