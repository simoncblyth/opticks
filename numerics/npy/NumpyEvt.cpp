#include "NumpyEvt.hpp"

#include "uif.h"
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Timer.hpp"
#include "stringutil.hpp"

#include "limits.h"
#include "assert.h"
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* NumpyEvt::genstep = "genstep" ; 
const char* NumpyEvt::photon  = "photon" ; 
const char* NumpyEvt::record  = "record" ; 
const char* NumpyEvt::phosel = "phosel" ; 
const char* NumpyEvt::recsel  = "recsel" ; 
const char* NumpyEvt::sequence  = "sequence" ; 
const char* NumpyEvt::aux = "aux" ; 


void NumpyEvt::init()
{
    m_timer = new Timer("NumpyEvt"); 
    m_timer->setVerbose(false);
}


ViewNPY* NumpyEvt::operator [](const char* spec)
{
    std::vector<std::string> elem ; 
    split(elem, spec, '.');

    if(elem.size() != 2 ) assert(0);

    MultiViewNPY* mvn(NULL); 
    if(     elem[0] == genstep) mvn = m_genstep_attr ;  
    else if(elem[0] == photon)  mvn = m_photon_attr ;
    else if(elem[0] == record)  mvn = m_record_attr ;
    else if(elem[0] == phosel)  mvn = m_phosel_attr ;
    else if(elem[0] == recsel)  mvn = m_recsel_attr ;
    else if(elem[0] == sequence)  mvn = m_sequence_attr ;
    else if(elem[0] == aux)      mvn = m_aux_attr ;

    assert(mvn);
    return (*mvn)[elem[1].c_str()] ;
}

void NumpyEvt::setGenstepData(NPY<float>* genstep)
{
    m_timer->start();

    m_genstep_data = genstep  ;

    m_genstep_attr = new MultiViewNPY();
    //                                                    j k sz   type        norm   iatt
    m_genstep_attr->add(new ViewNPY("vpos",m_genstep_data,1,0,4,ViewNPY::FLOAT,false,false));    // (x0, t0)                     2nd GenStep quad 
    m_genstep_attr->add(new ViewNPY("vdir",m_genstep_data,2,0,4,ViewNPY::FLOAT,false,false));    // (DeltaPosition, step_length) 3rd GenStep quad

    // attribute offset calulated by  npy->getByteIndex(0,j,k) 
    // assuming the size of the attribute type matches that of the NPY<T>

    m_num_gensteps = m_genstep_data->getShape(0) ;
    m_num_photons = m_genstep_data->getUSum(0,3);

    if(!m_optix)
    {
        LOG(info) << "NumpyEvt::setGenstepData early exit due to --nooptix/-O " ;
        return ;  
    } 

    createHostBuffers();

    //if(m_allocate)
    //{
    //    allocateHostBuffers();
    //    seedPhotonData();
    //}
    //else
    //{
    //    LOG(warning) << "NumpyEvt::setGenstepData skipping allocateHostBuffer/seedPhotonData : GPU alternatives need to be done " ;
    //}

    m_timer->stop();
    //m_timer->dump();
}

void NumpyEvt::createHostBuffers()
{
    LOG(debug) << "NumpyEvt::createHostBuffers "
              << " num_photons " << m_num_photons  
               ;

    NPY<float>* pho = NPY<float>::make(m_num_photons, 4, 4); // must match GPU side photon.h:PNUMQUAD
    setPhotonData(pho);   

    NPY<Sequence_t>* seq = NPY<Sequence_t>::make(m_num_photons, 1, 2);  // shape (np,1,2) (formerly initialized to 0)
    setSequenceData(seq);   

    NPY<unsigned char>* phosel = NPY<unsigned char>::make(m_num_photons,1,4); // shape (np,1,4) (formerly initialized to 0)
    setPhoselData(phosel);   

    unsigned int num_records = getNumRecords();

    NPY<short>* rec = NPY<short>::make(num_records, 2, 4);  // shape (nr,2,4) formerly initialized to SHRT_MIN
    setRecordData(rec);   

    // aka seqidx (SequenceNPY) or target ThrustIndex
    NPY<unsigned char>* recsel = NPY<unsigned char>::make(num_records,1,4); // shape (nr,1,4) (formerly initialized to 0) 
    setRecselData(recsel);   

    NPY<short>* aux = NPY<short>::make(num_records, 1, 4);  // shape (nr,1,4)
    setAuxData(aux);   


    (*m_timer)("createHostBuffers");
}

void NumpyEvt::allocateHostBuffers()
{
    // NPY::make(ni,nj,nk) now defaults to not allocating, 
    // so allocate here 
    //
    // TODO: eliminate this preallocation by moving to GPU resident


    LOG(info) << "NumpyEvt::allocateHostBuffers "
              << " num_photons " << m_num_photons  
               ;

    assert(0);

    m_photon_data->zero();
    m_photon_data->setAllowPrealloc(true);


  /*
    m_sequence_data->zero();
    m_sequence_data->setAllowPrealloc(true);

    m_phosel_data->zero();
    m_phosel_data->setAllowPrealloc(true);

    assert(SHRT_MIN == -(1 << 15));      // -32768
    assert(SHRT_MAX ==  (1 << 15) - 1);  // +32767

    m_record_data->fill(SHRT_MIN);
    m_record_data->setAllowPrealloc(true);

    m_recsel_data->zero();
    m_recsel_data->setAllowPrealloc(true);
  */

    (*m_timer)("allocateHostBuffers");
}




void NumpyEvt::seedPhotonData()
{
    //
    // NB cf with 
    //           Scene::uploadEvt
    //           Scene::uploadSelection
    //           OptiXEngine::init
    //
    // stuff genstep index into the photon allocation 
    // to allow generation to access appropriate genstep 
    //
    // see thrustrap-/iexpand.h and iexpandTest.cc for an 
    // initial try at moving genstep identification per photon
    // to GPU side  
    //

    G4StepNPY gs(m_genstep_data);  

    unsigned int numStep   = m_genstep_data->getShape(0);
    unsigned int numPhoton = m_photon_data->getShape(0);
    assert(numPhoton == m_num_photons);

    unsigned int count(0) ;
    for(unsigned int index=0 ; index < numStep ; index++)
    {
        unsigned int npho = m_genstep_data->getUInt(index, 0, 3);
        if(gs.isCerenkovStep(index))
        {
            //assert(npho > 0 && npho < 150); // by observation of Cerenkov steps
            assert(npho > 0 && npho < 3000);  
        }
        else if(gs.isScintillationStep(index))
        {
            assert(npho >= 0 && npho < 5000);     // by observation of Scintillation steps                  
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
    (*m_timer)("seedPhotonData");
}


void NumpyEvt::setPhotonData(NPY<float>* photon_data)
{
    m_photon_data = photon_data  ;
    m_photon_data->setDynamic();  // need to update with seeding so GL_DYNAMIC_DRAW needed 
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

void NumpyEvt::setAuxData(NPY<short>* aux_data)
{
    m_aux_data = aux_data  ;
    m_aux_attr = new MultiViewNPY();
    //                                            j k sz   type                  norm   iatt
    ViewNPY* ibnd = new ViewNPY("ibnd",m_aux_data,0,0,4,ViewNPY::SHORT          ,false,  true);
    m_aux_attr->add(ibnd);
}

void NumpyEvt::setRecordData(NPY<short>* record_data)
{
    m_record_data = record_data  ;

    //                                               j k sz   type                  norm   iatt
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,4,ViewNPY::SHORT          ,true,  false);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,4,ViewNPY::UNSIGNED_BYTE  ,true,  false);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,2,ViewNPY::UNSIGNED_SHORT ,false, true);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,1,2,4,ViewNPY::UNSIGNED_BYTE  ,false, true);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo
    

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
    m_record_attr->add(rflq);

}



void NumpyEvt::setPhoselData(NPY<unsigned char>* phosel_data)
{
    m_phosel_data = phosel_data ;
    //                                               j k sz   type                norm   iatt
    ViewNPY* psel = new ViewNPY("psel",m_phosel_data,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true);
    m_phosel_attr = new MultiViewNPY();
    m_phosel_attr->add(psel);
}


void NumpyEvt::setRecselData(NPY<unsigned char>* recsel_data)
{
    m_recsel_data = recsel_data ;
    //                                                  j k sz   type                norm   iatt
    ViewNPY* rsel = new ViewNPY("rsel",m_recsel_data,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true);
    m_recsel_attr = new MultiViewNPY();
    m_recsel_attr->add(rsel);
}


void NumpyEvt::setSequenceData(NPY<Sequence_t>* sequence_data)
{
    m_sequence_data = sequence_data  ;
    assert(sizeof(Sequence_t) == 4*sizeof(unsigned short));  
    //
    // 64 bit uint used to hold the sequence flag sequence 
    // is presented to OpenGL shaders as 4 *16bit ushort 
    // as intend to reuse the sequence bit space for the indices and count 
    // via some diddling 
    //
    //      Have not taken the diddling route, 
    //      instead using separate Recsel/Phosel buffers for the indices
    // 
    //                                                j k sz   type                norm   iatt
    ViewNPY* phis = new ViewNPY("phis",m_sequence_data,0,0,4,ViewNPY::UNSIGNED_SHORT,false,  true);
    ViewNPY* pmat = new ViewNPY("pmat",m_sequence_data,0,1,4,ViewNPY::UNSIGNED_SHORT,false,  true);
    m_sequence_attr = new MultiViewNPY();
    m_sequence_attr->add(phis);
    m_sequence_attr->add(pmat);

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



