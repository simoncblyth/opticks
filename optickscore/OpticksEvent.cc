
// optickscore-
#include "OpticksEvent.hh"
#include "Indexer.hh"


// npy-
#include "uif.h"
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Parameters.hpp"
#include "GLMFormat.hpp"
#include "Index.hpp"

#include "stringutil.hpp"
#include "timeutil.hpp"

#include "Report.hpp"
#include "Timer.hpp"
#include "Times.hpp"
#include "TimesTable.hpp"

#include "limits.h"
#include "assert.h"
#include <sstream>

#include "NLog.hpp"


#define TIMER(s) \
    { \
       if(m_timer)\
       {\
          Timer& t = *(m_timer) ;\
          t((s)) ;\
       }\
    }





const char* OpticksEvent::TIMEFORMAT = "%Y%m%d_%H%M%S" ;
const char* OpticksEvent::PARAMETERS_NAME = "parameters.json" ;

std::string OpticksEvent::timestamp()
{
    char* tsl =  now(TIMEFORMAT, 20, 0);
    std::string timestamp =  tsl ;
    free((void*)tsl);
    return timestamp ; 
}

const char* OpticksEvent::incoming = "incoming" ; 
const char* OpticksEvent::primary = "primary" ; 
const char* OpticksEvent::genstep = "genstep" ; 
const char* OpticksEvent::nopstep = "nopstep" ; 
const char* OpticksEvent::photon  = "photon" ; 
const char* OpticksEvent::record  = "record" ; 
const char* OpticksEvent::phosel = "phosel" ; 
const char* OpticksEvent::recsel  = "recsel" ; 
const char* OpticksEvent::sequence  = "sequence" ; 
const char* OpticksEvent::aux = "aux" ; 

void OpticksEvent::init()
{
    m_timer = new Timer("OpticksEvent"); 
    m_timer->setVerbose(false);
    m_timer->start();

    m_parameters = new Parameters ;
    m_report = new Report ; 

    m_parameters->add<std::string>("TimeStamp", timestamp() );
    m_parameters->add<std::string>("Type", m_typ );
    m_parameters->add<std::string>("Tag", m_tag );
    m_parameters->add<std::string>("Detector", m_det );
    m_parameters->add<std::string>("Cat", m_cat );
    m_parameters->add<std::string>("UDet", getUDet() );

    m_data_names.push_back(genstep);
    m_data_names.push_back(incoming);
    m_data_names.push_back(primary);
    m_data_names.push_back(nopstep);
    m_data_names.push_back(photon);
    m_data_names.push_back(record);
    m_data_names.push_back(aux);
    m_data_names.push_back(phosel);
    m_data_names.push_back(recsel);
    m_data_names.push_back(sequence);

    m_abbrev[genstep] = "" ;      // no-prefix : cerenkov or scintillation
    m_abbrev[incoming] = "in" ;   // not yet used
    m_abbrev[primary] = "pr" ;    // not yet used
    m_abbrev[nopstep] = "no" ;    // non optical particle steps obtained from G4 eg with g4gun
    m_abbrev[photon] = "ox" ;     // photon final step uncompressed 
    m_abbrev[record] = "rx" ;     // photon step compressed record
    m_abbrev[aux] = "au" ;        // formerly used for photon level debug 
    m_abbrev[phosel] = "ps" ;     // photon selection index
    m_abbrev[recsel] = "rs" ;     // record selection index
    m_abbrev[sequence] = "ph" ;   // (unsigned long long) photon seqhis/seqmat
}


NPYBase* OpticksEvent::getData(const char* name)
{
    NPYBase* data = NULL ; 
    if(     strcmp(name, genstep)==0) data = static_cast<NPYBase*>(m_genstep_data) ; 
    else if(strcmp(name, incoming)==0) data = static_cast<NPYBase*>(m_incoming_data) ;
    else if(strcmp(name, primary)==0) data = static_cast<NPYBase*>(m_primary_data) ;
    else if(strcmp(name, nopstep)==0) data = static_cast<NPYBase*>(m_nopstep_data) ;
    else if(strcmp(name, photon)==0)  data = static_cast<NPYBase*>(m_photon_data) ;
    else if(strcmp(name, record)==0)  data = static_cast<NPYBase*>(m_record_data) ;
    else if(strcmp(name, aux)==0)  data = static_cast<NPYBase*>(m_aux_data) ;
    else if(strcmp(name, phosel)==0)  data = static_cast<NPYBase*>(m_phosel_data) ;
    else if(strcmp(name, recsel)==0)  data = static_cast<NPYBase*>(m_recsel_data) ;
    else if(strcmp(name, sequence)==0) data = static_cast<NPYBase*>(m_sequence_data) ;
    return data ; 
}

std::string OpticksEvent::getShapeString()
{
    std::stringstream ss ; 
    for(std::vector<std::string>::const_iterator it=m_data_names.begin() ; it != m_data_names.end() ; it++)
    {
         std::string name = *it ; 
         NPYBase* data = getData(name.c_str());
         ss << " " << name << " " << ( data ? data->getShapeString() : "NULL" )  ; 
    }
    return ss.str();
}

std::string OpticksEvent::getTimeStamp()
{
    return m_parameters->get<std::string>("TimeStamp");
}
unsigned int OpticksEvent::getBounceMax()
{
    return m_parameters->get<unsigned int>("BounceMax");
}
unsigned int OpticksEvent::getRngMax()
{
    return m_parameters->get<unsigned int>("RngMax");
}


ViewNPY* OpticksEvent::operator [](const char* spec)
{
    std::vector<std::string> elem ; 
    split(elem, spec, '.');

    if(elem.size() != 2 ) assert(0);

    MultiViewNPY* mvn(NULL); 
    if(     elem[0] == genstep)  mvn = m_genstep_attr ;  
    else if(elem[0] == nopstep)  mvn = m_nopstep_attr ;
    else if(elem[0] == photon)   mvn = m_photon_attr ;
    else if(elem[0] == record)   mvn = m_record_attr ;
    else if(elem[0] == phosel)   mvn = m_phosel_attr ;
    else if(elem[0] == recsel)   mvn = m_recsel_attr ;
    else if(elem[0] == sequence) mvn = m_sequence_attr ;
    else if(elem[0] == aux)      mvn = m_aux_attr ;

    assert(mvn);
    return (*mvn)[elem[1].c_str()] ;
}


void OpticksEvent::setGenstepData(NPY<float>* genstep)
{
    m_genstep_data = genstep  ;
    m_parameters->add<std::string>("genstepDigest",   genstep->getDigestString()  );

    //                                                j k l sz   type        norm   iatt
    ViewNPY* vpos = new ViewNPY("vpos",m_genstep_data,1,0,0,4,ViewNPY::FLOAT,false,false);    // (x0, t0)                     2nd GenStep quad 
    ViewNPY* vdir = new ViewNPY("vdir",m_genstep_data,2,0,0,4,ViewNPY::FLOAT,false,false);    // (DeltaPosition, step_length) 3rd GenStep quad

    m_genstep_attr = new MultiViewNPY("genstep_attr");
    m_genstep_attr->add(vpos);
    m_genstep_attr->add(vdir);

    // attribute offset calulated by  npy->getByteIndex(0,j,k) 
    // assuming the size of the attribute type matches that of the NPY<T>

    {
        m_num_gensteps = m_genstep_data->getShape(0) ;
        unsigned int num_photons = m_genstep_data->getUSum(0,3);
        setNumPhotons(num_photons);

        createHostBuffers();
        if(m_step)
        {
            createHostIndexBuffers();
        }

        m_parameters->add<unsigned int>("NumGensteps", getNumGensteps());
        m_parameters->add<unsigned int>("NumPhotons",  getNumPhotons());

        if(m_step)
        {
            m_parameters->add<unsigned int>("NumRecords",  getNumRecords());
        }
    }
   
}

void OpticksEvent::resizeIndices()
{
    // needed for G4 loaded photon indexing 

    unsigned int num_photons = getNumPhotons();
    unsigned int num_records = getNumRecords();
    unsigned int num_phosel = m_phosel_data->getShape(0);
    unsigned int num_recsel = m_recsel_data->getShape(0);

    LOG(info) << "OpticksEvent::resizeIndices"
              << " num_photons " << num_photons  
              << " num_records " << num_records
              << " num_phosel " << num_phosel  
              << " num_recsel " << num_recsel
              ;

    assert(num_photons > 0 );
    assert(num_records > 0 );

    if(num_phosel != num_photons)
    {
        m_phosel_data->setNumItems(num_photons);
        LOG(warning) << "OpticksEvent::resizeIndices changed phosel items from " << num_phosel << " to " << num_photons ; 
    }
    if(num_recsel != num_records)
    {
        m_recsel_data->setNumItems(num_records);
        LOG(warning) << "OpticksEvent::resizeIndices changed recsel items from " << num_recsel << " to " << num_records ; 
    }
}

void OpticksEvent::prepareForIndexing()
{
    if(!m_step) return ; 
    assert(m_num_photons > 0 );

    createHostIndexBuffers();
}

void OpticksEvent::prepareForPrimaryRecording()
{
    LOG(info) << "OpticksEvent::prepareForPrimaryRecording"
              << " m_num_photons " << m_num_photons  
               ;

   NPY<float>* primary = NPY<float>::make(m_num_photons, 4, 4) ;
   setPrimaryData(primary);
}

void OpticksEvent::createBuffers()
{
    // CPU running does not have the CUDA thread
    // not knowing the order problem, so it is possible
    // to not know the allocation ahead of time

    createHostBuffers();
    createHostIndexBuffers();
}

void OpticksEvent::createHostBuffers()
{
    // NB this does not allocate much memory, the NPY just hold
    // the shapes. Allocations are triggered by zero-ing the buffers. 
    //  
    // CFG4 CPU Geant4 running creates:
    //     photon, record, sequence buffers that can be just loaded 
    //

    (*m_timer)("_createHostBuffers");

    unsigned int num_photons = getNumPhotons();
    unsigned int num_records = getNumRecords();

    LOG(info) << "OpticksEvent::createHostBuffers "
              << " flat " << m_flat 
              << " num_photons " << num_photons  
              << " num_records " << num_records  
              << " maxrec " << m_maxrec
               ;

    createPhotonBuffers(num_photons);

    if(m_flat)
        createFlatRecordBuffers(num_records);
    else
        createStructuredRecordBuffers(num_photons, m_maxrec);

    createDomainBuffers();

    LOG(info) << "OpticksEvent::createHostBuffers DONE " ;

    (*m_timer)("createHostBuffers");
}


void OpticksEvent::createHostIndexBuffers()
{
    assert( m_step );

    // this unceremoniously replaces/leaks prior buffers...

    unsigned int num_photons = getNumPhotons();
    unsigned int num_records = getNumRecords();

    LOG(info) << "OpticksEvent::createHostIndexBuffers "
              << " flat " << m_flat 
              << " num_photons " << num_photons  
              << " num_records " << num_records 
              << " m_maxrec " << m_maxrec
               ;

    NPY<unsigned char>* phosel = NPY<unsigned char>::make(num_photons,1,4); // shape (np,1,4) (formerly initialized to 0)
    setPhoselData(phosel);   

    NPY<unsigned char>* recsel = NULL ; 
    if(m_flat)
        recsel = NPY<unsigned char>::make(num_records,1,4); // shape (nr,1,4) (formerly initialized to 0) 
    else
        recsel = NPY<unsigned char>::make(num_photons, m_maxrec,1,4); // shape (nr,1,4) (formerly initialized to 0) 

    setRecselData(recsel);   
}


void OpticksEvent::createPhotonBuffers(unsigned int num_photons)
{
    NPY<float>* pho = NPY<float>::make(num_photons, 4, 4); // must match GPU side photon.h:PNUMQUAD
    setPhotonData(pho);   

    NPY<unsigned long long>* seq = NPY<unsigned long long>::make(num_photons, 1, 2);  // shape (np,1,2) (formerly initialized to 0)
    setSequenceData(seq);   
}

void OpticksEvent::createFlatRecordBuffers(unsigned int num_records)
{
    if(m_step)
    {
        NPY<short>* rec = NPY<short>::make(num_records, 2, 4);  // shape (nr,2,4) formerly initialized to SHRT_MIN
        setRecordData(rec);   

        //NPY<unsigned char>* recsel = NPY<unsigned char>::make(num_records,1,4); // shape (nr,1,4) (formerly initialized to 0) 
        //setRecselData(recsel);   
    }

    NPY<short>* aux = NPY<short>::make(num_records, 1, 4);  // shape (nr,1,4)
    setAuxData(aux);   
}

void OpticksEvent::createStructuredRecordBuffers(unsigned int num_photons, unsigned int maxrec)
{
    if(m_step)
    {
        NPY<short>* rec = NPY<short>::make(num_photons, maxrec, 2, 4); 
        setRecordData(rec);   

        //NPY<unsigned char>* recsel = NPY<unsigned char>::make(num_photons, maxrec,1,4); // shape (nr,1,4) (formerly initialized to 0) 
        //setRecselData(recsel);   
    }

    NPY<short>* aux = NPY<short>::make(num_photons, maxrec, 1, 4);  // shape (nr,1,4)
    setAuxData(aux);   
}

void OpticksEvent::createDomainBuffers()
{
    NPY<float>* fdom = NPY<float>::make(3,1,4);
    setFDomain(fdom);

    NPY<int>* idom = NPY<int>::make(1,1,4);
    setIDomain(idom);

    // these small ones can be zeroed directly 
    fdom->zero();
    idom->zero();
}





void OpticksEvent::dumpDomains(const char* msg)
{
    LOG(info) << msg 
              << "\n space_domain      " << gformat(m_space_domain)
              << "\n time_domain       " << gformat(m_time_domain)
              << "\n wavelength_domain " << gformat(m_wavelength_domain)
              ;
}

void OpticksEvent::updateDomainsBuffer()
{
    NPY<float>* fdom = getFDomain();
    if(fdom)
    {
        fdom->setQuad(m_space_domain     , 0);
        fdom->setQuad(m_time_domain      , 1);
        fdom->setQuad(m_wavelength_domain, 2);
    }
    else
    {
        LOG(warning) << "OpticksEvent::updateDomainsBuffer fdom NULL " ;
    }


    NPY<int>* idom = getIDomain();

    if(idom)
        idom->setQuad(m_settings, 0 );
    else
        LOG(warning) << "OpticksEvent::updateDomainsBuffer idom NULL " ;
    
}

void OpticksEvent::readDomainsBuffer()
{
    NPY<float>* fdom = getFDomain();

    if(fdom)
    {
        m_space_domain = fdom->getQuad(0);
        m_time_domain = fdom->getQuad(1);
        m_wavelength_domain = fdom->getQuad(2);
    }
    else
    {
        LOG(warning) << "OpticksEvent::readDomainsBuffer"
                     << " fdom NULL "
                     ;
    }


    NPY<int>* idom = getIDomain();

    if(idom)
    {
        m_settings = idom->getQuad(0); 
        m_maxrec = m_settings.w ; 

        LOG(info) << "OpticksEvent::readDomainsBuffer" 
                  << " from idom settings m_maxrec " << m_maxrec 
                  ;
    }
    else
    {
        LOG(warning) << "OpticksEvent::readDomainsBuffer"
                     << " idom NULL "
                     ;
 
    }
 

}



void OpticksEvent::zero()
{
    if(m_photon_data)
        m_photon_data->zero();
    else
        LOG(warning) << "OpticksEvent::zero NULL photon_data " ;

    if(m_aux_data) m_aux_data->zero();

    if(m_step)
    {
        if(m_phosel_data)   m_phosel_data->zero();
        if(m_record_data)   m_record_data->zero();
        if(m_recsel_data)   m_recsel_data->zero();
        if(m_sequence_data) m_sequence_data->zero();
    }
}



void OpticksEvent::seedPhotonData()
{
    assert(0); // this is now done by opop-/OpSeeder

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
            m_photon_data->setUInt(count, 0,0,0, index );  // set "phead" : repeating step index for every photon to be generated for the step
            count += 1 ;         
        }  // over photons for each step
    }      // over gen steps


    LOG(info) << "OpticksEvent::setGenstepData " 
              << " stepId(0) " << gs.getStepId(0) 
              << " genstep length " << numStep 
              << " photon length " << numPhoton
              << "  num_photons " << m_num_photons  ; 

    assert(count == m_num_photons ); 
    assert(count == numPhoton ); 
    // not m_num_photons-1 as last incremented count value is not used by setUInt
    (*m_timer)("seedPhotonData");
}


void OpticksEvent::setPhotonData(NPY<float>* photon_data)
{
    m_photon_data = photon_data  ;
    if(m_num_photons == 0) 
    {
        m_num_photons = photon_data->getShape(0) ;

        LOG(info) << "OpticksEvent::setPhotonData"
                  << " setting m_num_photons from shape(0) " << m_num_photons 
                  ;
    }
    else
    {
        assert(m_num_photons == photon_data->getShape(0));
    }

    m_photon_data->setDynamic();  // need to update with seeding so GL_DYNAMIC_DRAW needed 
    m_photon_attr = new MultiViewNPY("photon_attr");
    //                                                  j k l,sz   type          norm   iatt
    m_photon_attr->add(new ViewNPY("vpos",m_photon_data,0,0,0,4,ViewNPY::FLOAT, false, false));      // 1st quad
    m_photon_attr->add(new ViewNPY("vdir",m_photon_data,1,0,0,4,ViewNPY::FLOAT, false, false));      // 2nd quad
    m_photon_attr->add(new ViewNPY("vpol",m_photon_data,2,0,0,4,ViewNPY::FLOAT, false, false));      // 3rd quad
    m_photon_attr->add(new ViewNPY("iflg",m_photon_data,3,0,0,4,ViewNPY::INT  , false, true ));      // 4th quad

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

void OpticksEvent::setAuxData(NPY<short>* aux_data)
{
    m_aux_data = aux_data  ;

    if(m_aux_data == NULL)
    {
       LOG(warning) << "OpticksEvent::setAuxData NULL";
       return ;  
    }

    m_aux_attr = new MultiViewNPY("aux_attr");
    //                                            j k l sz   type                  norm   iatt
    ViewNPY* ibnd = new ViewNPY("ibnd",m_aux_data,0,0,0,4,ViewNPY::SHORT          ,false,  true);
    m_aux_attr->add(ibnd);
}



void OpticksEvent::setNopstepData(NPY<float>* nopstep)
{
    m_nopstep_data = nopstep  ;
    if(!nopstep) return ; 

    m_num_nopsteps = m_nopstep_data->getShape(0) ;
    LOG(info) << "OpticksEvent::setNopstepData"
              << " shape " << nopstep->getShapeString()
              ;

    //                                                j k l sz   type         norm   iatt
    ViewNPY* vpos = new ViewNPY("vpos",m_nopstep_data,0,0,0,4,ViewNPY::FLOAT ,false,  false);
    ViewNPY* vdir = new ViewNPY("vdir",m_nopstep_data,1,0,0,4,ViewNPY::FLOAT ,false,  false);   
    ViewNPY* vpol = new ViewNPY("vpol",m_nopstep_data,2,0,0,4,ViewNPY::FLOAT ,false,  false);   

    m_nopstep_attr = new MultiViewNPY("nopstep_attr");
    m_nopstep_attr->add(vpos);
    m_nopstep_attr->add(vdir);
    m_nopstep_attr->add(vpol);

    // createHostBuffers();  
    // only allocates small buffers, big ones deferred til usage
}


void OpticksEvent::setRecordData(NPY<short>* record_data)
{
    assert(m_step);

    m_record_data = record_data  ;

    //                                               j k l sz   type                  norm   iatt
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0,4,ViewNPY::SHORT          ,true,  false);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,0,4,ViewNPY::UNSIGNED_BYTE  ,true,  false);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,0,2,ViewNPY::UNSIGNED_SHORT ,false, true);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,1,2,0,4,ViewNPY::UNSIGNED_BYTE  ,false, true);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo
    

    // ViewNPY::TYPE need not match the NPY<T>,
    // OpenGL shaders will view the data as of the ViewNPY::TYPE, 
    // informed via glVertexAttribPointer/glVertexAttribIPointer 
    // in oglrap-/Rdr::address(ViewNPY* vnpy)
 
    // standard byte offsets obtained from from sizeof(T)*value_offset 
    //rpol->setCustomOffset(sizeof(unsigned char)*rpol->getValueOffset());
    // this is not needed

    m_record_attr = new MultiViewNPY("record_attr");

    m_record_attr->add(rpos);
    m_record_attr->add(rpol);
    m_record_attr->add(rflg);
    m_record_attr->add(rflq);
}


void OpticksEvent::setPhoselData(NPY<unsigned char>* phosel_data)
{
    assert(m_step);
    m_phosel_data = phosel_data ;
    if(!m_phosel_data) return ; 

    //                                               j k l sz   type                norm   iatt
    ViewNPY* psel = new ViewNPY("psel",m_phosel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true);
    m_phosel_attr = new MultiViewNPY("phosel_attr");
    m_phosel_attr->add(psel);
}


void OpticksEvent::setRecselData(NPY<unsigned char>* recsel_data)
{
    assert(m_step);
    m_recsel_data = recsel_data ;
    if(!m_recsel_data) return ; 
    //                                               j k l sz   type                norm   iatt
    ViewNPY* rsel = new ViewNPY("rsel",m_recsel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true);
    m_recsel_attr = new MultiViewNPY("recsel_attr");
    m_recsel_attr->add(rsel);
}


void OpticksEvent::setSequenceData(NPY<unsigned long long>* sequence_data)
{
    assert(m_step);
    m_sequence_data = sequence_data  ;
    assert(sizeof(unsigned long long) == 4*sizeof(unsigned short));  
    //
    // 64 bit uint used to hold the sequence flag sequence 
    // is presented to OpenGL shaders as 4 *16bit ushort 
    // as intend to reuse the sequence bit space for the indices and count 
    // via some diddling 
    //
    //      Have not taken the diddling route, 
    //      instead using separate Recsel/Phosel buffers for the indices
    // 
    //                                                 j k l sz   type                norm   iatt
    ViewNPY* phis = new ViewNPY("phis",m_sequence_data,0,0,0,4,ViewNPY::UNSIGNED_SHORT,false,  true);
    ViewNPY* pmat = new ViewNPY("pmat",m_sequence_data,0,1,0,4,ViewNPY::UNSIGNED_SHORT,false,  true);
    m_sequence_attr = new MultiViewNPY("sequence_attr");
    m_sequence_attr->add(phis);
    m_sequence_attr->add(pmat);

}


void OpticksEvent::dumpPhotonData()
{
    if(!m_photon_data) return ;
    dumpPhotonData(m_photon_data);
}

void OpticksEvent::dumpPhotonData(NPY<float>* photons)
{
    std::cout << photons->description("OpticksEvent::dumpPhotonData") << std::endl ;

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



void OpticksEvent::Summary(const char* msg)
{
    LOG(info) << description(msg) ; 
}

std::string OpticksEvent::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " 
       << " typ: " << m_typ 
       << " tag: " << m_tag 
       << " det: " << m_det 
       << " cat: " << m_cat 
       << " udet: " << getUDet()
       << " num_photons: " <<  m_num_photons
       ;

    //if(m_genstep_data)  ss << m_genstep_data->description("m_genstep_data") ;
    //if(m_photon_data)   ss << m_photon_data->description("m_photon_data") ;

    return ss.str();
}


void OpticksEvent::recordDigests()
{

    NPY<float>* ox = getPhotonData() ;
    if(ox && ox->hasData())
        m_parameters->add<std::string>("photonData",   ox->getDigestString()  );

    NPY<short>* au = getAuxData() ;
    if(au && au->hasData())
        m_parameters->add<std::string>("auxData",      au->getDigestString()  );

    if(m_step)
    {
        NPY<short>* rx = getRecordData() ;
        if(rx && rx->hasData())
            m_parameters->add<std::string>("recordData",   rx->getDigestString()  );

        NPY<unsigned long long>* ph = getSequenceData() ;
        if(ph && ph->hasData())
            m_parameters->add<std::string>("sequenceData", ph->getDigestString()  );
    }
}

void OpticksEvent::save(bool verbose)
{
    (*m_timer)("_save");

    recordDigests();

    const char* udet = getUDet();
    LOG(info) << "OpticksEvent::save"
              << " typ: " << m_typ
              << " tag: " << m_tag
              << " det: " << m_det
              << " cat: " << m_cat
              << " udet: " << udet 
              ;    

    LOG(info) << "OpticksEvent::save " << getShapeString() ; 

   // genstep normally not saved as it exists already coming from elsewhere,
   // but for TorchStep that insnt the case

    NPY<float>* pr = getPrimaryData();
    if(pr)
    {
        pr->setVerbose(verbose);
        pr->save("pr%s", m_typ,  m_tag, udet);
    }

    NPY<float>* no = getNopstepData();
    if(no)
    {
        no->setVerbose(verbose);
        no->save("no%s", m_typ,  m_tag, udet);
        no->dump("OpticksEvent::save (nopstep)");
    }

    NPY<float>* ox = getPhotonData();
    if(ox)
    {
        ox->setVerbose(verbose);
        ox->save("ox%s", m_typ,  m_tag, udet);
    } 

    NPY<short>* rx = getRecordData();    
    if(rx)
    {
        rx->setVerbose(verbose);
        rx->save("rx%s", m_typ,  m_tag, udet);
    }

    NPY<unsigned long long>* ph = getSequenceData();
    if(ph)
    {
        ph->setVerbose(verbose);
        ph->save("ph%s", m_typ,  m_tag, udet);
    }

    NPY<short>* au = getAuxData();
    if(au && au->hasData())
    {
        au->setVerbose(verbose);
        au->save("au%s", m_typ,  m_tag, udet);
    } 

    updateDomainsBuffer();

    NPY<float>* fdom = getFDomain();
    if(fdom) fdom->save("fdom%s", m_typ,  m_tag, udet);

    NPY<int>* idom = getIDomain();
    if(idom) idom->save("idom%s", m_typ,  m_tag, udet);

    if(no)
    {
       assert(idom && "OpticksEvent::save non-null nopstep BUT HAS NULL IDOM ");
    }


    saveIndex(verbose);
    saveParameters();

    (*m_timer)("save");

    makeReport();  // after timer save, in order to include that in the report
    saveReport();
}



void OpticksEvent::saveIndex(bool verbose)
{
    const char* udet = getUDet();

    NPY<unsigned char>* ps = getPhoselData();
    if(ps)
    {
        ps->setVerbose(verbose);
        ps->save("ps%s", m_typ,  m_tag, udet);
    }

    NPY<unsigned char>* rs = getRecselData();
    if(rs)
    {
        rs->setVerbose(verbose);
        rs->save("rs%s", m_typ,  m_tag, udet);
    }

    std::string ixdir = getSpeciesDir("ix");
    LOG(info) << "OpticksEvent::saveIndex"
              << " ixdir " << ixdir
              << " seqhis " << m_seqhis
              << " seqmat " << m_seqmat
              << " bndidx " << m_bndidx
              ; 
   

    if(m_seqhis)
        m_seqhis->save(ixdir.c_str(), m_tag);        
    else
        LOG(warning) << "OpticksEvent::saveIndex no seqhis to save " ;

    if(m_seqmat)
        m_seqmat->save(ixdir.c_str(), m_tag);        
    else
        LOG(warning) << "OpticksEvent::saveIndex no seqmat to save " ;

    if(m_bndidx)
        m_bndidx->save(ixdir.c_str(), m_tag);        
    else
        LOG(warning) << "OpticksEvent::saveIndex no bndidx to save " ;

}


void OpticksEvent::loadIndex()
{
    std::string ixdir = getSpeciesDir("ix");
    // TODO: promote OpticksEvent into Opticks OR OpOp in order to have access to the opticks- header for this
    m_seqhis = Index::load(ixdir.c_str(), m_tag, "History_Sequence" );  // SEQHIS_NAME_
    m_seqmat = Index::load(ixdir.c_str(), m_tag, "Material_Sequence");  // SEQMAT_NAME_
    m_bndidx = Index::load(ixdir.c_str(), m_tag, "Boundary_Index");     // BNDIDX_NAME_
}



void OpticksEvent::makeReport()
{
    LOG(info) << "OpticksEvent::makeReport" ; 

    m_parameters->dump();

    m_timer->stop();

    m_ttable = m_timer->makeTable();
    m_ttable->dump("OpticksEvent::makeReport");

    m_report->add(m_parameters->getLines());
    m_report->add(m_ttable->getLines());
}


std::string OpticksEvent::speciesDir(const char* species, const char* udet, const char* typ)
{
    std::string dir = NPYBase::directory(species, typ, udet );
    return dir ; 
}


std::string OpticksEvent::getSpeciesDir(const char* species)
{
    const char* udet = getUDet();
    return speciesDir(species, udet, m_typ );
}

std::string OpticksEvent::getTagDir(const char* species, bool tstamp)
{
    std::stringstream ss ;
    ss << getSpeciesDir(species) << "/" << m_tag  ;
    if(tstamp) ss << "/" << getTimeStamp() ;
    return ss.str();
}


void OpticksEvent::saveParameters()
{
    std::string mddir = getTagDir("md", false);
    m_parameters->save(mddir.c_str(), PARAMETERS_NAME);

    std::string mddir_ts = getTagDir("md", true);
    m_parameters->save(mddir_ts.c_str(), PARAMETERS_NAME);
}


void OpticksEvent::loadParameters()
{
    std::string pmdir = getTagDir("md", false);
    m_parameters->load_(pmdir.c_str(), PARAMETERS_NAME );
}

void OpticksEvent::saveReport()
{
    std::string mdd = getTagDir("md", false);  
    saveReport(mdd.c_str());

    std::string mdd_ts = getTagDir("md", true);  
    saveReport(mdd_ts.c_str());
}



void OpticksEvent::saveReport(const char* dir)
{
    if(!m_ttable || !m_report) return ; 
    LOG(info) << "OpticksEvent::saveReport to " << dir  ; 

    m_ttable->save(dir);
    m_report->save(dir);  
}

void OpticksEvent::loadReport()
{
    std::string mdd = getTagDir("md", false);  
    m_ttable = Timer::loadTable(mdd.c_str());
    m_report = Report::load(mdd.c_str());
}

void OpticksEvent::setFakeNopstepPath(const char* path)
{
    // fake path used by OpticksEvent::load rather than standard one
    // see npy-/nopstep_viz_debug.py

    m_fake_nopstep_path = path ? strdup(path) : NULL ;
}


OpticksEvent* OpticksEvent::load(const char* typ, const char* tag, const char* det, const char* cat, bool verbose)
{
    OpticksEvent* evt = new OpticksEvent(typ, tag, det, cat);
    evt->loadBuffers(verbose);
    if(evt->isNoLoad())
    {
         LOG(warning) << "OpticksEvent::load FAILED " ;
         delete evt ;
         evt = NULL ;
    } 
    return evt ;  
}


void OpticksEvent::loadBuffers(bool verbose)
{
    (*m_timer)("_load");
    const char* udet = strlen(m_cat) > 0 ? m_cat : m_det ; 

    bool qload = true ; 
    NPY<int>*   idom = NPY<int>::load("idom%s", m_typ,  m_tag, udet, qload);
    if(!idom)
    {
        m_noload = true ; 
        LOG(warning) << "OpticksEvent::load NO SUCH EVENT : RUN WITHOUT --load OPTION TO CREATE IT " 
                     << " typ: " << m_typ
                     << " tag: " << m_tag
                     << " det: " << m_det
                     << " cat: " << m_cat
                     << " udet: " << udet 
                    ;     
        return ; 
    }

    m_loaded = true ; 

    NPY<float>* fdom = NPY<float>::load("fdom%s", m_typ,  m_tag, udet, qload );

    setIDomain(idom);
    setFDomain(fdom);

    loadReport();
    loadParameters();
    loadIndex();

    readDomainsBuffer();
    dumpDomains("OpticksEvent::load dumpDomains");

    NPY<float>* no = NULL ; 
    if(m_fake_nopstep_path)
    {
        LOG(warning) << "OpticksEvent::load using setFakeNopstepPath " << m_fake_nopstep_path ; 
        no = NPY<float>::debugload(m_fake_nopstep_path);
    }
    else
    {  
        no = NPY<float>::load("no%s", m_typ,  m_tag, udet, qload);
    }



    NPY<float>* pr = NPY<float>::load("pr%s", m_typ,  m_tag, udet, qload);
    NPY<float>* ox = NPY<float>::load("ox%s", m_typ,  m_tag, udet, qload);
    NPY<short>* au = NPY<short>::load("au%s", m_typ,  m_tag, udet, qload);
    
    NPY<unsigned long long>* ph = NULL ; 
    NPY<short>*              rx = NULL ; 
    NPY<unsigned char>*      ps = NULL ; 
    NPY<unsigned char>*      rs = NULL ; 

    // hmm should m_step be detected from the files or imposed from config  
    if(m_step)
    {
        rx = NPY<short>::load("rx%s", m_typ,  m_tag, udet, qload);
        ph = NPY<unsigned long long>::load("ph%s", m_typ,  m_tag, udet, qload );
        ps = NPY<unsigned char>::load("ps%s", m_typ,  m_tag, udet, qload );
        rs = NPY<unsigned char>::load("rs%s", m_typ,  m_tag, udet, qload );
    }

    unsigned int num_nopstep = no ? no->getShape(0) : 0 ;
    unsigned int num_photons = ox ? ox->getShape(0) : 0 ;
    unsigned int num_history = ph ? ph->getShape(0) : 0 ;
    unsigned int num_phosel  = ps ? ps->getShape(0) : 0 ;

    // either zero or matching 
    assert(num_history == 0 || num_photons == num_history );
    assert(num_phosel == 0 || num_photons == num_phosel );

    unsigned int num_records = rx ? rx->getShape(0) : 0 ;
    unsigned int num_aux     = au ? au->getShape(0) : 0 ;
    unsigned int num_recsel  = rs ? rs->getShape(0) : 0 ;

    assert(num_records == 0 || num_aux == 0 || num_records == num_aux ); 
    assert(num_recsel == 0 || num_records == num_recsel );


    LOG(info) << "OpticksEvent::load shape(0) before reshaping "
              << " num_nopstep " << num_nopstep
              << " [ "
              << " num_photons " << num_photons
              << " num_history " << num_history
              << " num_phosel " << num_phosel 
              << " ] "
              << " [ "
              << " num_records " << num_records
              << " num_recsel " << num_recsel
              << " num_aux " << num_aux 
              << " ] "
              ; 


    if(num_records == num_photons*m_maxrec)
    {
        LOG(info) << "OpticksEvent::load flat records (Opticks style) detected " ;
        setFlat(true);
    } 
    else if(num_records == num_photons)
    {
        LOG(info) << "OpticksEvent::load structured records (cfg4- style) detected :  RESHAPING " ;
        if(rx && num_records > 0)
        {
            if(verbose) rx->Summary("rx init");
            unsigned int ni = rx->getShape(0);
            unsigned int nj = rx->getShape(1);
            unsigned int nk = rx->getShape(2);
            unsigned int nl = rx->getShape(3);
            rx->reshape(ni*nj, nk, nl, 0);
            if(verbose) rx->Summary("rx reshaped");
        }       
        
        if(rs && num_recsel > 0)
        {
            if(verbose) rs->Summary("rs init");
            unsigned int ni = rs->getShape(0);
            unsigned int nj = rs->getShape(1);
            unsigned int nk = rs->getShape(2);
            unsigned int nl = rs->getShape(3);
            rs->reshape(ni*nj, nk, nl, 0);
            if(verbose) rs->Summary("rs reshaped");
        }       
        if(au && num_aux > 0)
        {
            if(verbose) au->Summary("au init");
            unsigned int ni = au->getShape(0);
            unsigned int nj = au->getShape(1);
            unsigned int nk = au->getShape(2);
            unsigned int nl = au->getShape(3);
            au->reshape(ni*nj, nk, nl, 0);       
            if(verbose) au->Summary("au reshaped");
        }
        setFlat(true);
    }
    else
    {
         LOG(info) << "OpticksEvent::load no step " ; 
    }



    setNopstepData(no);
    setPrimaryData(pr);
    setPhotonData(ox);
    setSequenceData(ph);
    setRecordData(rx);
    setAuxData(au);

    setPhoselData(ps);
    setRecselData(rs);

    (*m_timer)("load");


    LOG(info) << "OpticksEvent::load " << getShapeString() ; 

    if(verbose)
    {
        fdom->Summary("fdom");
        idom->Summary("idom");

        if(no) no->Summary("no");
        if(ox) ox->Summary("ox");
        if(rx) rx->Summary("rx");
        if(ph) ph->Summary("ph");
        if(au) au->Summary("au");
        if(ps) ps->Summary("ps");
        if(rs) rs->Summary("rs");
    }

}

bool OpticksEvent::isIndexed()
{
    return m_phosel_data != NULL && m_recsel_data != NULL && m_seqhis != NULL && m_seqmat != NULL ;
}



NPY<float>* OpticksEvent::loadGenstepDerivativeFromFile(const char* postfix, bool quietly)
{
    char tag[128];
    snprintf(tag, 128, "%s_%s", m_tag, postfix );


    if(!quietly)
    LOG(info) << "OpticksEvent::loadGenstepDerivativeFromFile  "
              << " typ " << m_typ
              << " tag " << tag
              << " det " << m_det
              ;

    NPY<float>* npy = NPY<float>::load(m_typ, tag, m_det, quietly) ;
    if(npy)
    {
        npy->dump("OpticksEvent::loadGenstepDerivativeFromFile");
    }
    else
    {
        if(!quietly)
        LOG(warning) << "OpticksEvent::loadGenstepDerivativeFromFile FAILED for postfix " << postfix ;
    }
    return npy ; 
}


NPY<float>* OpticksEvent::loadGenstepFromFile(int modulo)
{
    LOG(info) << "OpticksEvent::loadGenstepFromFile  "
              << " typ " << m_typ
              << " tag " << m_tag
              << " det " << m_det
              ;

    NPY<float>* npy = NPY<float>::load(m_typ, m_tag, m_det ) ;

    m_parameters->add<std::string>("genstepAsLoaded",   npy->getDigestString()  );

    m_parameters->add<int>("Modulo", modulo );

    if(modulo > 0)
    {
        LOG(warning) << "App::loadGenstepFromFile applying modulo scaledown " << modulo ;
        npy = NPY<float>::make_modulo(npy, modulo);
        m_parameters->add<std::string>("genstepModulo",   npy->getDigestString()  );
    }
    return npy ;
}





void OpticksEvent::setNumG4Event(unsigned int n)
{
   m_parameters->add<int>("NumG4Event", n);
}
void OpticksEvent::setNumPhotonsPerG4Event(unsigned int n)
{
   m_parameters->add<int>("NumPhotonsPerG4Event", n);
}
unsigned int OpticksEvent::getNumG4Event()
{
   return m_parameters->get<int>("NumG4Event");
}
unsigned int OpticksEvent::getNumPhotonsPerG4Event()
{
   return m_parameters->get<int>("NumPhotonsPerG4Event");
}
 

void OpticksEvent::indexPhotonsCPU()
{
    // see tests/IndexerTest

    LOG(info) << "OpticksEvent::indexPhotonsCPU" ; 

    NPY<unsigned long long>* sequence = getSequenceData();
    NPY<unsigned char>*        phosel = getPhoselData();
    assert(sequence->getShape(0) == phosel->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence();
    idx->applyLookup<unsigned char>(phosel->getValues());

    // idx->repeat_to( m_maxrec ...)
    // TODO: phosel->recsel by repeating by maxrec

    setHistoryIndex(idx->getHistoryIndex());
    setMaterialIndex(idx->getMaterialIndex());

    TIMER("indexPhotonsCPU");    
}


