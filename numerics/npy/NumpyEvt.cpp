#include "NumpyEvt.hpp"

#include "uif.h"
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

#include "limits.h"
#include "assert.h"
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void NumpyEvt::setGenstepData(NPY<float>* genstep)
{
    G4StepNPY gs(genstep);  

    m_genstep_data = genstep  ;
    m_genstep_attr = new MultiViewNPY();
    //                                                    j k sz   type        norm   iatt
    m_genstep_attr->add(new ViewNPY("vpos",m_genstep_data,1,0,4,ViewNPY::FLOAT,false,false));    // (x0, t0)                     2nd GenStep quad 
    m_genstep_attr->add(new ViewNPY("vdir",m_genstep_data,2,0,4,ViewNPY::FLOAT,false,false));    // (DeltaPosition, step_length) 3rd GenStep quad

    // attribute offset calulated by  npy->getByteIndex(0,j,k) 
    // assuming the size of the attribute type matches that of the NPY<T>

    m_num_photons = m_genstep_data->getUSum(0,3);

    NPY<float>* pho = NPY<float>::make_vec4(m_num_photons, 4); // must match GPU side photon.h:PNUMQUAD
    setPhotonData(pho);   




    assert(SHRT_MIN == -(1 << 15));      // -32768
    assert(SHRT_MAX ==  (1 << 15) - 1);  // +32767
    NPY<short>* rec = NPY<short>::make_vec4(getNumRecords(), 2, SHRT_MIN); 
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
    m_photon_attr = new MultiViewNPY();
    //                                                  j k sz   type          norm   iatt
    m_photon_attr->add(new ViewNPY("vpos",m_photon_data,0,0,4,ViewNPY::FLOAT, false, false));      // 1st quad
    m_photon_attr->add(new ViewNPY("vdir",m_photon_data,1,0,4,ViewNPY::FLOAT, false, false));      // 2nd quad
    m_photon_attr->add(new ViewNPY("vpol",m_photon_data,2,0,4,ViewNPY::FLOAT, false, false));      // 3rd quad
    m_photon_attr->add(new ViewNPY("iflg",m_photon_data,3,0,4,ViewNPY::INT  , false, true ));      // 4th quad

    //
    //  photon array 
    //  ~~~~~~~~~~~~~
    //     
    //  vpos  xxxx yyyy zzzz wwww    position, time           [:,0,:4]
    //  vdir  xxxx yyyy zzzz wwww    direction, wavelength    [:,1,:4]
    //  vpol  xxxx yyyy zzzz wwww    polarization weight      [:,2,:4] 
    //  iflg  xxxx yyyy zzzz wwww                             [:,3,:4]
    //
    //
    //  record array
    //  ~~~~~~~~~~~~~~
    //       
    //              4*short(snorm)
    //          ________
    //  rpos    xxyyzzww 
    //  rpol->  xyzwaabb <-rflg 
    //          ----^^^^
    //     4*ubyte     2*ushort   
    //     (unorm)     (iatt)
    //
    //
    //
    // corresponds to GPU side cu/photon.h:psave and rsave 
    //
}

void NumpyEvt::setRecordData(NPY<short>* record_data)
{
    m_record_data = record_data  ;

    //                                               j k sz   type                  norm   iatt
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,4,ViewNPY::SHORT          ,true,  false);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,4,ViewNPY::UNSIGNED_BYTE  ,true,  false);   
    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,2,ViewNPY::UNSIGNED_SHORT ,false, true);   

    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  
    //
    // ViewNPY::TYPE need not match the NPY<T>,
    // OpenGL shaders will view the data as of the ViewNPY::TYPE, 
    // informed via glVertexAttribPointer/glVertexAttribIPointer 
    // in oglrap-/Rdr::address(ViewNPY* vnpy)
 



    // standard byte offsets obtained from from sizeof(T)*value_offset 
    //rpol->setCustomOffset(sizeof(unsigned char)*rpol->getValueOffset());
    // this is not needed

    m_record_attr = new MultiViewNPY();
    m_record_attr->add(rpos);
    m_record_attr->add(rpol);
    m_record_attr->add(rflg);

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



