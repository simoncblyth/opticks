#ifdef _MSC_VER
// 'ViewNPY': object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


#include <climits>
#include <cassert>
#include <sstream>
#include <cstring>
#include <iomanip>

// sysrap-
#include "STimes.hh"

// brap-
#include "BStr.hh"
#include "BTime.hh"
#include "BFile.hh"
#include "BOpticksResource.hh"
#include "BOpticksEvent.hh"

// npy-
#include "uif.h"
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NLoad.hpp"
#include "NPYSpec.hpp"
#include "NLookup.hpp"

#include "G4StepNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Parameters.hpp"
#include "GLMFormat.hpp"
#include "Index.hpp"

#include "Report.hpp"
#include "Timer.hpp"
#include "Times.hpp"
#include "TimesTable.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksSwitches.h"
#include "OpticksPhoton.h"
#include "OpticksConst.hh"
#include "OpticksDomain.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "OpticksMode.hh"
#include "OpticksBufferControl.hh"
#include "OpticksActionControl.hh"
#include "Indexer.hh"

#include "PLOG.hh"


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
const char* OpticksEvent::PARAMETERS_STEM = "parameters" ;
const char* OpticksEvent::PARAMETERS_EXT = ".json" ;


std::string OpticksEvent::timestamp()
{
    std::string timestamp = BTime::now(TIMEFORMAT, 0);
    return timestamp ; 
}

const char* OpticksEvent::fdom_    = "fdom" ; 
const char* OpticksEvent::idom_    = "idom" ; 
const char* OpticksEvent::genstep_ = "genstep" ; 
const char* OpticksEvent::nopstep_ = "nopstep" ; 
const char* OpticksEvent::photon_  = "photon" ; 
const char* OpticksEvent::record_  = "record" ; 
const char* OpticksEvent::phosel_ = "phosel" ; 
const char* OpticksEvent::recsel_  = "recsel" ; 
const char* OpticksEvent::sequence_  = "sequence" ; 
const char* OpticksEvent::seed_  = "seed" ; 


OpticksEvent* OpticksEvent::make(OpticksEventSpec* spec, unsigned tagoffset)
{
     OpticksEventSpec* offspec = spec->clone(tagoffset);
     return new OpticksEvent(offspec) ; 
}

OpticksEvent::OpticksEvent(OpticksEventSpec* spec) 
          :
          OpticksEventSpec(spec),
          m_event_spec(spec),
          m_ok(NULL),   // set by Opticks::makeEvent

          m_noload(false),
          m_loaded(false),

          m_timer(NULL),
          m_parameters(NULL),
          m_report(NULL),
          m_ttable(NULL),

          m_primary_data(NULL),
          m_genstep_data(NULL),
          m_nopstep_data(NULL),
          m_photon_data(NULL),
          m_record_data(NULL),
          m_phosel_data(NULL),
          m_recsel_data(NULL),
          m_sequence_data(NULL),
          m_seed_data(NULL),

          m_photon_ctrl(NULL),
          m_domain(NULL),

          m_g4step(NULL),
          m_genstep_vpos(NULL),
          m_genstep_attr(NULL),
          m_nopstep_attr(NULL),
          m_photon_attr(NULL),
          m_record_attr(NULL),
          m_phosel_attr(NULL),
          m_recsel_attr(NULL),
          m_sequence_attr(NULL),
          m_seed_attr(NULL),

          m_records(NULL),
          m_photons(NULL),
          m_hits(NULL),
          m_bnd(NULL),

          m_num_gensteps(0),
          m_num_nopsteps(0),
          m_num_photons(0),

          m_seqhis(NULL),
          m_seqmat(NULL),
          m_bndidx(NULL),
          m_fake_nopstep_path(NULL),

          m_fdom_spec(NULL),
          m_idom_spec(NULL),
          m_genstep_spec(NULL),
          m_nopstep_spec(NULL),
          m_photon_spec(NULL),
          m_record_spec(NULL),
          m_phosel_spec(NULL),
          m_recsel_spec(NULL),
          m_sequence_spec(NULL),

          m_prelaunch_times(new STimes),
          m_launch_times(new STimes),
          m_sibling(NULL)
{
    init();
}


OpticksEvent::~OpticksEvent()
{
    LOG(info) << "OpticksEvent::~OpticksEvent PLACEHOLDER" ; 
} 


STimes* OpticksEvent::getPrelaunchTimes()
{
    return m_prelaunch_times ; 
}
STimes* OpticksEvent::getLaunchTimes()
{
    return m_launch_times ; 
}

void OpticksEvent::setSibling(OpticksEvent* sibling)
{
    m_sibling = sibling ; 
}
OpticksEvent* OpticksEvent::getSibling()
{
    return m_sibling ; 
}





bool OpticksEvent::isNoLoad()
{
    return m_noload ; 
}
bool OpticksEvent::isLoaded()
{
    return m_loaded ; 
}
bool OpticksEvent::isStep()
{
    return true  ; 
}
bool OpticksEvent::isFlat()
{
    return false  ; 
}




unsigned int OpticksEvent::getNumGensteps()
{
    return m_num_gensteps ; 
}
unsigned int OpticksEvent::getNumNopsteps()
{
    return m_num_nopsteps ; 
}

void OpticksEvent::resizeToZero()
{
    bool resize = true ; 
    setNumPhotons(0, resize);
}

void OpticksEvent::setNumPhotons(unsigned int num_photons, bool resize_)
{
    m_num_photons = num_photons ; 
    if(resize_)
    {
        LOG(trace) << "OpticksEvent::setNumPhotons RESIZING " << num_photons ;  
        resize();
    }
    else
    {
        LOG(trace) << "OpticksEvent::setNumPhotons NOT RESIZING " << num_photons ;  
    }
}
unsigned int OpticksEvent::getNumPhotons()
{
    return m_num_photons ; 
}


unsigned int OpticksEvent::getNumRecords()
{
    unsigned int maxrec = getMaxRec();
    return m_num_photons * maxrec ; 
}
unsigned int OpticksEvent::getMaxRec()
{
    return m_domain->getMaxRec() ; 
}
void OpticksEvent::setMaxRec(unsigned int maxrec)
{
    m_domain->setMaxRec(maxrec);
}



bool OpticksEvent::hasGenstepData()
{
    return m_genstep_data && m_genstep_data->hasData() ; 
}
bool OpticksEvent::hasPhotonData()
{
    return m_photon_data && m_photon_data->hasData() ; 
}




NPY<float>* OpticksEvent::getGenstepData()
{ 
     return m_genstep_data ;
}
NPY<float>* OpticksEvent::getNopstepData() 
{ 
     return m_nopstep_data ; 
}
NPY<float>* OpticksEvent::getPhotonData()
{
     return m_photon_data ; 
} 
NPY<short>* OpticksEvent::getRecordData()
{ 
    return m_record_data ; 
}
NPY<unsigned char>* OpticksEvent::getPhoselData()
{ 
    return m_phosel_data ;
}
NPY<unsigned char>* OpticksEvent::getRecselData()
{ 
    return m_recsel_data ; 
}
NPY<unsigned long long>* OpticksEvent::getSequenceData()
{ 
    return m_sequence_data ;
}
NPY<unsigned>* OpticksEvent::getSeedData()
{ 
    return m_seed_data ;
}


MultiViewNPY* OpticksEvent::getGenstepAttr(){ return m_genstep_attr ; }
MultiViewNPY* OpticksEvent::getNopstepAttr(){ return m_nopstep_attr ; }
MultiViewNPY* OpticksEvent::getPhotonAttr(){ return m_photon_attr ; }
MultiViewNPY* OpticksEvent::getRecordAttr(){ return m_record_attr ; }
MultiViewNPY* OpticksEvent::getPhoselAttr(){ return m_phosel_attr ; }
MultiViewNPY* OpticksEvent::getRecselAttr(){ return m_recsel_attr ; }
MultiViewNPY* OpticksEvent::getSequenceAttr(){ return m_sequence_attr ; }
MultiViewNPY* OpticksEvent::getSeedAttr(){  return m_seed_attr ; }



void OpticksEvent::setRecordsNPY(RecordsNPY* records)
{
    m_records = records ; 
}
RecordsNPY* OpticksEvent::getRecordsNPY()
{
    return m_records ;
}

void OpticksEvent::setPhotonsNPY(PhotonsNPY* photons)
{
    m_photons = photons ; 
}
PhotonsNPY* OpticksEvent::getPhotonsNPY()
{
    return m_photons ;
}


void OpticksEvent::setHitsNPY(HitsNPY* hits)
{
    m_hits = hits ; 
}
HitsNPY* OpticksEvent::getHitsNPY()
{
    return m_hits ;
}


void OpticksEvent::setBoundariesNPY(BoundariesNPY* bnd)
{
    m_bnd = bnd ; 
}
BoundariesNPY* OpticksEvent::getBoundariesNPY()
{
    return m_bnd ;
}









void OpticksEvent::setFDomain(NPY<float>* fdom)
{
    m_domain->setFDomain(fdom) ; 
}
void OpticksEvent::setIDomain(NPY<int>* idom)
{
    m_domain->setIDomain(idom) ; 
}

NPY<float>* OpticksEvent::getFDomain()
{
    return m_domain->getFDomain() ; 
}
NPY<int>* OpticksEvent::getIDomain()
{
    return m_domain->getIDomain() ; 
}


// below set by Opticks::makeEvent

void OpticksEvent::setMode(OpticksMode* mode)
{ 
    m_mode = mode ; 
}
void OpticksEvent::setSpaceDomain(const glm::vec4& space_domain)
{
    m_domain->setSpaceDomain(space_domain) ; 
}
void OpticksEvent::setTimeDomain(const glm::vec4& time_domain)
{
    m_domain->setTimeDomain(time_domain)  ; 
}
void OpticksEvent::setWavelengthDomain(const glm::vec4& wavelength_domain)
{
    m_domain->setWavelengthDomain(wavelength_domain)  ; 
}


bool OpticksEvent::isInterop()
{
    return m_mode->isInterop();
}
bool OpticksEvent::isCompute()
{
    return m_mode->isCompute();
}

const glm::vec4& OpticksEvent::getSpaceDomain()
{
    return m_domain->getSpaceDomain() ; 
}
const glm::vec4& OpticksEvent::getTimeDomain()
{
    return m_domain->getTimeDomain() ;
}
const glm::vec4& OpticksEvent::getWavelengthDomain()
{ 
    return m_domain->getWavelengthDomain() ; 
}





void OpticksEvent::setBoundaryIndex(Index* bndidx)
{
    // called from OpIndexer::indexBoundaries
    m_bndidx = bndidx ; 
}
void OpticksEvent::setHistoryIndex(Index* seqhis)
{
    // called from OpIndexer::indexSequenceLoaded 
    m_seqhis = seqhis ; 
}
void OpticksEvent::setMaterialIndex(Index* seqmat)
{
    // called from OpIndexer::indexSequenceLoaded
    m_seqmat = seqmat ; 
}


Index* OpticksEvent::getHistoryIndex()
{
    return m_seqhis ; 
} 
Index* OpticksEvent::getMaterialIndex()
{
    return m_seqmat ; 
} 
Index* OpticksEvent::getBoundaryIndex()
{
    return m_bndidx ; 
}




Parameters* OpticksEvent::getParameters()
{
    return m_parameters ;
}
Timer* OpticksEvent::getTimer()
{
    return m_timer ;
}
TimesTable* OpticksEvent::getTimesTable()
{
    return m_ttable ;
}


void OpticksEvent::init()
{
    m_timer = new Timer("OpticksEvent"); 
    m_timer->setVerbose(false);
    m_timer->start();

    m_parameters = new Parameters ;
    m_report = new Report ; 
    m_domain = new OpticksDomain ; 

    m_parameters->add<std::string>("TimeStamp", timestamp() );
    m_parameters->add<std::string>("Type", m_typ );
    m_parameters->add<std::string>("Tag", m_tag );
    m_parameters->add<std::string>("Detector", m_det );
    if(m_cat) m_parameters->add<std::string>("Cat", m_cat );
    m_parameters->add<std::string>("UDet", getUDet() );

    m_data_names.push_back(genstep_);
    m_data_names.push_back(nopstep_);
    m_data_names.push_back(photon_);
    m_data_names.push_back(record_);
    m_data_names.push_back(phosel_);
    m_data_names.push_back(recsel_);
    m_data_names.push_back(sequence_);
    m_data_names.push_back(seed_);

    m_abbrev[genstep_] = "gs" ;    // input gs are named: cerenkov, scintillation but for posterity need common output tag
    m_abbrev[nopstep_] = "no" ;    // non optical particle steps obtained from G4 eg with g4gun
    m_abbrev[photon_] = "ox" ;     // photon final step uncompressed 
    m_abbrev[record_] = "rx" ;     // photon step compressed record
    m_abbrev[phosel_] = "ps" ;     // photon selection index
    m_abbrev[recsel_] = "rs" ;     // record selection index
    m_abbrev[sequence_] = "ph" ;   // (unsigned long long) photon seqhis/seqmat
    m_abbrev[seed_] = "se" ;   //   (short) genstep id used for photon seeding 
}


NPYBase* OpticksEvent::getData(const char* name)
{
    NPYBase* data = NULL ; 
    if(     strcmp(name, genstep_)==0) data = static_cast<NPYBase*>(m_genstep_data) ; 
    else if(strcmp(name, nopstep_)==0) data = static_cast<NPYBase*>(m_nopstep_data) ;
    else if(strcmp(name, photon_)==0)  data = static_cast<NPYBase*>(m_photon_data) ;
    else if(strcmp(name, record_)==0)  data = static_cast<NPYBase*>(m_record_data) ;
    else if(strcmp(name, phosel_)==0)  data = static_cast<NPYBase*>(m_phosel_data) ;
    else if(strcmp(name, recsel_)==0)  data = static_cast<NPYBase*>(m_recsel_data) ;
    else if(strcmp(name, sequence_)==0) data = static_cast<NPYBase*>(m_sequence_data) ;
    else if(strcmp(name, seed_)==0) data = static_cast<NPYBase*>(m_seed_data) ;
    return data ; 
}

NPYSpec* OpticksEvent::getSpec(const char* name)
{
    NPYSpec* spec = NULL ; 
    if(     strcmp(name, genstep_)==0) spec = static_cast<NPYSpec*>(m_genstep_spec) ; 
    else if(strcmp(name, nopstep_)==0) spec = static_cast<NPYSpec*>(m_nopstep_spec) ;
    else if(strcmp(name, photon_)==0)  spec = static_cast<NPYSpec*>(m_photon_spec) ;
    else if(strcmp(name, record_)==0)  spec = static_cast<NPYSpec*>(m_record_spec) ;
    else if(strcmp(name, phosel_)==0)  spec = static_cast<NPYSpec*>(m_phosel_spec) ;
    else if(strcmp(name, recsel_)==0)  spec = static_cast<NPYSpec*>(m_recsel_spec) ;
    else if(strcmp(name, sequence_)==0) spec = static_cast<NPYSpec*>(m_sequence_spec) ;
    else if(strcmp(name, seed_)==0)     spec = static_cast<NPYSpec*>(m_seed_spec) ;
    else if(strcmp(name, fdom_)==0)     spec = static_cast<NPYSpec*>(m_fdom_spec) ;
    else if(strcmp(name, idom_)==0)     spec = static_cast<NPYSpec*>(m_idom_spec) ;
    return spec ; 
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


void OpticksEvent::setOpticks(Opticks* ok)
{
    m_ok = ok ; 
}

int OpticksEvent::getId()
{
    return m_parameters->get<int>("Id");
}
void OpticksEvent::setId(int id)
{
    m_parameters->add<int>("Id", id);
}

void OpticksEvent::setCreator(const char* executable)
{
    m_parameters->add<std::string>("Creator", executable ? executable : "NULL" );
}
std::string OpticksEvent::getCreator()
{
    return m_parameters->get<std::string>("Creator", "NONE");
}

void OpticksEvent::setTimeStamp(const char* tstamp)
{
    m_parameters->set<std::string>("TimeStamp", tstamp);
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
    return m_parameters->get<unsigned int>("RngMax", "0");
}
void OpticksEvent::setRngMax(unsigned int rng_max)
{
    m_parameters->add<unsigned int>("RngMax",    rng_max );
}



ViewNPY* OpticksEvent::operator [](const char* spec)
{
    std::vector<std::string> elem ; 
    BStr::split(elem, spec, '.');

    if(elem.size() != 2 ) assert(0);

    MultiViewNPY* mvn(NULL); 
    if(     elem[0] == genstep_)  mvn = m_genstep_attr ;  
    else if(elem[0] == nopstep_)  mvn = m_nopstep_attr ;
    else if(elem[0] == photon_)   mvn = m_photon_attr ;
    else if(elem[0] == record_)   mvn = m_record_attr ;
    else if(elem[0] == phosel_)   mvn = m_phosel_attr ;
    else if(elem[0] == recsel_)   mvn = m_recsel_attr ;
    else if(elem[0] == sequence_) mvn = m_sequence_attr ;
    else if(elem[0] == seed_)     mvn = m_seed_attr ;

    assert(mvn);
    return (*mvn)[elem[1].c_str()] ;
}

/*


   INTEROP mode GPU buffer access C:create R:read W:write
   ----------------------------------------------------------

                 OpenGL     OptiX              Thrust 

   gensteps       CR       R (gen/prop)       R (seeding)

   photons        CR       W (gen/prop)       W (seeding)
   sequence                W (gen/prop)
   phosel         CR                          W (indexing) 

   records        CR       W  
   recsel         CR                          W (indexing)


   OptiX has no business with phosel and recsel 

*/


NPYSpec* OpticksEvent::GenstepSpec()
{
    return new NPYSpec(genstep_   ,  0,6,4,0,      NPYBase::FLOAT     , "OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY")  ;
}

NPYSpec* OpticksEvent::SeedSpec()
{
    return new NPYSpec(seed_     ,  0,1,1,0,      NPYBase::UINT      , "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY,UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY") ;
}

void OpticksEvent::createSpec()
{
    // invoked by Opticks::makeEvent   or OpticksEvent::load
    unsigned int maxrec = getMaxRec();

    m_genstep_spec = GenstepSpec();

    m_seed_spec    = SeedSpec();

#ifdef WITH_SEED_BUFFER
    m_photon_spec   = new NPYSpec(photon_   ,  0,4,4,0,      NPYBase::FLOAT     , "OPTIX_OUTPUT_ONLY,INTEROP_PTR_FROM_OPENGL") ;
#else
    m_photon_spec   = new NPYSpec(photon_   ,  0,4,4,0,      NPYBase::FLOAT     , "OPTIX_INPUT_OUTPUT,INTEROP_PTR_FROM_OPENGL,BUFFER_COPY_ON_DIRTY") ;
          //   OPTIX_INPUT_OUTPUT : INPUT needed as seeding writes genstep identifiers into photon buffer
#endif


          //     INTEROP_PTR_FROM_OPENGL  : needed with OptiX 4.0, as OpenGL/OptiX/CUDA 3-way interop no longer working 
          //                        instead move to 
          //                                 OpenGL/OptiX : to write the photon data
          //                                 OpenGL/CUDA  : to index the photons  

    m_record_spec   = new NPYSpec(record_   ,  0,maxrec,2,4, NPYBase::SHORT     , "OPTIX_OUTPUT_ONLY") ;
         
          //   SHORT -> RT_FORMAT_SHORT4 and size set to  num_quads = num_photons*maxrec*2  

    m_sequence_spec = new NPYSpec(sequence_ ,  0,1,2,0,      NPYBase::ULONGLONG , "OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY") ;

          // OPTIX_NON_INTEROP  : creates OptiX buffer even in INTEROP mode, this is possible for sequence as 
          //                      it is not used by OpenGL shaders so no need for INTEROP
          //
          //    ULONGLONG -> RT_FORMAT_USER  and size set to ni*nj*nk = num_photons*1*2


    m_nopstep_spec = new NPYSpec(nopstep_   ,  0,4,4,0,      NPYBase::FLOAT     , "" ) ;
    m_fdom_spec    = new NPYSpec(fdom_      ,  3,1,4,0,      NPYBase::FLOAT     , "" ) ;
    m_idom_spec    = new NPYSpec(idom_      ,  1,1,4,0,      NPYBase::INT       , "" ) ;

        // OptiX buffers never created for nopstep, fdom, idom  

    m_phosel_spec   = new NPYSpec(phosel_   ,  0,1,4,0,      NPYBase::UCHAR     , "" ) ;
    m_recsel_spec   = new NPYSpec(recsel_   ,  0,maxrec,1,4, NPYBase::UCHAR     , "" ) ;

         // OptiX never sees phosel or recsel, they are written by Thrust by application of the index
         // and are read by OpenGL shaders to do record (and photon) selection
}


void OpticksEvent::addBufferControl(const char* name, const char* ctrl_)
{
    NPYBase* npy = getData(name);
    assert(npy);

    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    ctrl.add(ctrl_);

    LOG(info) << "OpticksEvent::addBufferControl"
              << " name " << name 
              << " adding " << ctrl_ 
              << " " << ctrl.description("result:") 
              ;

}


void OpticksEvent::setBufferControl(NPYBase* data)
{
    NPYSpec* spec = data->getBufferSpec();
    const char* name = data->getBufferName(); 

    if(!spec)
    {

        LOG(fatal) << "OpticksEvent::setBufferControl"
                     << " SKIPPED FOR " << name 
                     << " AS NO spec "
                     ;

        Parameters* param = data->getParameters();
        if(param)
            param->dump("OpticksEvent::setBufferControl FATAL: BUFFER LACKS SPEC"); 
        assert(0);
        return ; 
    }


    OpticksBufferControl ctrl(data->getBufferControlPtr());
    ctrl.add(spec->getCtrl());

    if(m_mode->isCompute()) ctrl.add(OpticksBufferControl::COMPUTE_MODE_) ; 
    if(m_mode->isInterop()) ctrl.add(OpticksBufferControl::INTEROP_MODE_) ; 

    if(ctrl("VERBOSE_MODE"))
     LOG(info) << std::setw(10) << name 
               << " : " << ctrl.description("(spec)") 
               ;
}



void OpticksEvent::createBuffers(NPY<float>* gs)
{
    // invoked by Opticks::makeEvent 

    // NB allocation is deferred until zeroing and they start at 0 items anyhow
    //
    // NB by default gs = false and genstep buffer creation is excluded, 
    //    those coming externally
    //    however they are needed for "template" zero events 
    //
    
    if(gs)   
    {
        bool progenitor = false ;    
        setGenstepData(gs, progenitor);   
    }

    NPY<float>* nop = NPY<float>::make(m_nopstep_spec); 
    setNopstepData(nop);   

    NPY<float>* pho = NPY<float>::make(m_photon_spec); // must match GPU side photon.h:PNUMQUAD
    setPhotonData(pho);   

    NPY<unsigned long long>* seq = NPY<unsigned long long>::make(m_sequence_spec); 
    setSequenceData(seq);   

    NPY<unsigned>* seed = NPY<unsigned>::make(m_seed_spec); 
    setSeedData(seed);   

    NPY<unsigned char>* phosel = NPY<unsigned char>::make(m_phosel_spec); 
    setPhoselData(phosel);   

    NPY<unsigned char>* recsel = NPY<unsigned char>::make(m_recsel_spec); 
    setRecselData(recsel);   

    NPY<short>* rec = NPY<short>::make(m_record_spec); 
    setRecordData(rec);   

    NPY<float>* fdom = NPY<float>::make(m_fdom_spec);
    setFDomain(fdom);

    NPY<int>* idom = NPY<int>::make(m_idom_spec);
    setIDomain(idom);

    // these small ones can be zeroed directly 
    fdom->zero();
    idom->zero();
}


void OpticksEvent::reset()
{
    resetBuffers();
}
void OpticksEvent::resetBuffers()
{
    if(m_nopstep_data)  m_nopstep_data->reset();    
    if(m_photon_data)   m_photon_data->reset();    
    if(m_sequence_data) m_sequence_data->reset();    
    if(m_seed_data)     m_seed_data->reset();    
    if(m_phosel_data)   m_phosel_data->reset();    
    if(m_recsel_data)   m_recsel_data->reset();    
    if(m_record_data)   m_record_data->reset();    
}


void OpticksEvent::resize()
{
    // NB these are all photon level qtys on the first dimension
    //    including recsel and record thanks to structured arrays (num_photons, maxrec, ...)

    assert(m_photon_data);
    assert(m_sequence_data);
    assert(m_phosel_data);
    assert(m_recsel_data);
    assert(m_record_data);
    assert(m_seed_data);

    unsigned int num_photons = getNumPhotons();
    unsigned int num_records = getNumRecords();
    unsigned int maxrec = getMaxRec();
 
    unsigned rng_max = getRngMax(); 
    bool enoughRng = num_photons <= rng_max ; 
    if(!enoughRng)
        LOG(fatal) << "OpticksEvent::resize  NOT ENOUGH RNG "
                   << " num_photons " << num_photons
                   << " rng_max " << rng_max 
                   ;
    assert(enoughRng && " need to prepare and persist more RNG states up to maximual per propagation number" );




    LOG(info) << "OpticksEvent::resize " 
              << " num_photons " << num_photons  
              << " num_records " << num_records 
              << " maxrec " << maxrec
              << " " << getDir()
              ;

    m_photon_data->setNumItems(num_photons);
    m_sequence_data->setNumItems(num_photons);
    m_seed_data->setNumItems(num_photons);
    m_phosel_data->setNumItems(num_photons);
    m_recsel_data->setNumItems(num_photons);
    m_record_data->setNumItems(num_photons);

    m_parameters->add<unsigned int>("NumGensteps", getNumGensteps());
    m_parameters->add<unsigned int>("NumPhotons",  getNumPhotons());
    m_parameters->add<unsigned int>("NumRecords",  getNumRecords());

}


void OpticksEvent::zero()
{
    if(m_photon_data)   m_photon_data->zero();
    if(m_sequence_data) m_sequence_data->zero();
    if(m_record_data)   m_record_data->zero();

    // when operating CPU side phosel and recsel are derived from sequence data
    // when operating GPU side they need not ever come to CPU
    //if(m_phosel_data)   m_phosel_data->zero();
    //if(m_recsel_data)   m_recsel_data->zero();
}


void OpticksEvent::dumpDomains(const char* msg)
{
    m_domain->dump(msg);
}
void OpticksEvent::updateDomainsBuffer()
{
    m_domain->updateBuffer();
}
void OpticksEvent::importDomainsBuffer()
{
    m_domain->importBuffer();
}


void OpticksEvent::setGenstepData(NPY<float>* genstep_data, bool progenitor, const char* oac_label)
{
    int nitems = NPYBase::checkNumItems(genstep_data);
    if(nitems < 1)
    {
         LOG(warning) << "OpticksEvent::setGenstepData SKIP "
                      << " nitems " << nitems
                      ;
         return ; 
    } 

    importGenstepData(genstep_data, oac_label );

    setBufferControl(genstep_data);

    m_genstep_data = genstep_data  ;
    m_parameters->add<std::string>("genstepDigest",   m_genstep_data->getDigestString()  );

    //                                                j k l sz   type        norm   iatt  item_from_dim
    ViewNPY* vpos = new ViewNPY("vpos",m_genstep_data,1,0,0,4,ViewNPY::FLOAT,false,false, 1);    // (x0, t0)                     2nd GenStep quad 
    ViewNPY* vdir = new ViewNPY("vdir",m_genstep_data,2,0,0,4,ViewNPY::FLOAT,false,false, 1);    // (DeltaPosition, step_length) 3rd GenStep quad

    m_genstep_vpos = vpos ; 

    m_genstep_attr = new MultiViewNPY("genstep_attr");
    m_genstep_attr->add(vpos);
    m_genstep_attr->add(vdir);

    {
        m_num_gensteps = m_genstep_data->getShape(0) ;
        unsigned int num_photons = m_genstep_data->getUSum(0,3);
        bool resize = progenitor ; 
        setNumPhotons(num_photons, resize); // triggers a resize   <<<<<<<<<<<<< SPECIAL HANDLING OF GENSTEP <<<<<<<<<<<<<<
    }
}

const glm::vec4& OpticksEvent::getGenstepCenterExtent()
{
    assert(m_genstep_vpos && "check hasGenstepData() before getGenstepCenterExtent"); 
    return m_genstep_vpos->getCenterExtent() ; 
}

G4StepNPY* OpticksEvent::getG4Step()
{
    return m_g4step ; 
}


void OpticksEvent::translateLegacyGensteps(NPY<float>* gs)
{
    OpticksActionControl oac(gs->getActionControlPtr());
    bool gs_torch = oac.isSet("GS_TORCH") ; 
    bool gs_legacy = oac.isSet("GS_LEGACY") ; 

    if(!gs_legacy) return ; 
    assert(!gs_torch); // there are no legacy torch files ?

    if(gs->isGenstepTranslated())
    {
        LOG(warning) << "OpticksEvent::translateLegacyGensteps already translated " ;
        return ; 
    }

    gs->setGenstepTranslated();

    NLookup* lookup = gs->getLookup();
    if(!lookup)
            LOG(fatal) << "OpticksEvent::translateLegacyGensteps"
                       << " IMPORT OF LEGACY GENSTEPS REQUIRES gs->setLookup(NLookup*) "
                       << " PRIOR TO OpticksEvent::setGenstepData(gs) "
                       ;

    assert(lookup); 

    m_g4step->relabel(CERENKOV, SCINTILLATION); 

    // CERENKOV or SCINTILLATION codes are used depending on 
    // the sign of the pre-label 
    // this becomes the ghead.i.x used in cu/generate.cu
    // which dictates what to generate

    lookup->close("OpticksEvent::translateLegacyGensteps GS_LEGACY");

    m_g4step->setLookup(lookup);   
    m_g4step->applyLookup(0, 2);  // jj, kk [1st quad, third value] is materialIndex
    // replaces original material indices with material lines
    // for easy access to properties using boundary_lookup GPU side

}


bool OpticksEvent::isTorchType()
{    
   return strcmp(m_typ, OpticksFlags::torch_) == 0 ; 
}
bool OpticksEvent::isMachineryType()
{    
   return strcmp(m_typ, OpticksFlags::machinery_) == 0 ; 
}


void OpticksEvent::importGenstepDataLoaded(NPY<float>* gs)
{
     OpticksActionControl ctrl(gs->getActionControlPtr());     
     ctrl.add(OpticksActionControl::GS_LOADED_);
     if(isTorchType())  ctrl.add(OpticksActionControl::GS_TORCH_);
}

void OpticksEvent::importGenstepData(NPY<float>* gs, const char* oac_label)
{
    Parameters* gsp = gs->getParameters();
    m_parameters->append(gsp);

    gs->setBufferSpec(OpticksEvent::GenstepSpec());

    assert(m_g4step == NULL && "OpticksEvent::importGenstepData can only do this once ");
    m_g4step = new G4StepNPY(gs);    

    OpticksActionControl oac(gs->getActionControlPtr());
    if(oac_label)
    {
        LOG(debug) << "OpticksEvent::importGenstepData adding oac_label " << oac_label ; 
        oac.add(oac_label);
    }


    LOG(debug) << "OpticksEvent::importGenstepData"
               << brief()
               << " shape " << gs->getShapeString()
               << " " << oac.description("oac")
               ;

    if(oac("GS_LEGACY"))
    {
        translateLegacyGensteps(gs);
    }
    else if(oac("GS_TORCH"))
    {
        LOG(debug) << " checklabel of torch steps  " << oac.description("oac") ; 
        m_g4step->checklabel(TORCH); 
    }
    else if(oac("GS_FABRICATED"))
    {
        m_g4step->checklabel(FABRICATED); 
    }
    else
    {
        LOG(debug) << " checklabel of non-legacy (collected direct) gensteps  " << oac.description("oac") ; 
        m_g4step->checklabel(CERENKOV, SCINTILLATION);
    }

    m_g4step->countPhotons();

    LOG(debug) 
         << " Keys "
         << " TORCH: " << TORCH 
         << " CERENKOV: " << CERENKOV 
         << " SCINTILLATION: " << SCINTILLATION  
         << " G4GUN: " << G4GUN  
         ;

     LOG(debug) 
         << " counts " 
         << m_g4step->description()
         ;
 

}









OpticksBufferControl* OpticksEvent::getPhotonCtrl()
{
   return m_photon_ctrl ; 
}

void OpticksEvent::setPhotonData(NPY<float>* photon_data)
{
    setBufferControl(photon_data);

    m_photon_data = photon_data  ;
    m_photon_ctrl = new OpticksBufferControl(m_photon_data->getBufferControlPtr());
    if(m_num_photons == 0) 
    {
        m_num_photons = photon_data->getShape(0) ;

        LOG(debug) << "OpticksEvent::setPhotonData"
                  << " setting m_num_photons from shape(0) " << m_num_photons 
                  ;
    }
    else
    {
        assert(m_num_photons == photon_data->getShape(0));
    }

    m_photon_data->setDynamic();  // need to update with seeding so GL_DYNAMIC_DRAW needed 
    m_photon_attr = new MultiViewNPY("photon_attr");
    //                                                  j k l,sz   type          norm   iatt  item_from_dim
    m_photon_attr->add(new ViewNPY("vpos",m_photon_data,0,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 1st quad
    m_photon_attr->add(new ViewNPY("vdir",m_photon_data,1,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 2nd quad
    m_photon_attr->add(new ViewNPY("vpol",m_photon_data,2,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 3rd quad
    m_photon_attr->add(new ViewNPY("iflg",m_photon_data,3,0,0,4,ViewNPY::INT  , false, true , 1));      // 4th quad

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



void OpticksEvent::setNopstepData(NPY<float>* nopstep)
{
    int nitems = NPYBase::checkNumItems(nopstep);
    if(nitems < 1)
    {
         LOG(debug) << "OpticksEvent::setNopstepData SKIP "
                      << " nitems " << nitems
                      ;
         return ; 
    } 



    setBufferControl(nopstep);

    m_nopstep_data = nopstep  ;
    if(!nopstep) return ; 

    m_num_nopsteps = m_nopstep_data->getShape(0) ;
    LOG(debug) << "OpticksEvent::setNopstepData"
              << " shape " << nopstep->getShapeString()
              ;

    //                                                j k l sz   type         norm   iatt   item_from_dim
    ViewNPY* vpos = new ViewNPY("vpos",m_nopstep_data,0,0,0,4,ViewNPY::FLOAT ,false,  false, 1);
    ViewNPY* vdir = new ViewNPY("vdir",m_nopstep_data,1,0,0,4,ViewNPY::FLOAT ,false,  false, 1);   
    ViewNPY* vpol = new ViewNPY("vpol",m_nopstep_data,2,0,0,4,ViewNPY::FLOAT ,false,  false, 1);   

    m_nopstep_attr = new MultiViewNPY("nopstep_attr");
    m_nopstep_attr->add(vpos);
    m_nopstep_attr->add(vdir);
    m_nopstep_attr->add(vpol);

}


void OpticksEvent::setRecordData(NPY<short>* record_data)
{
    setBufferControl(record_data);

    m_record_data = record_data  ;


#ifdef OLDWAY
    //                                               j k l  sz   type                  norm   iatt   item_from_dim
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0 ,4,ViewNPY::SHORT          ,true,  false, 2);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,0 ,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,0 ,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,1,2,0 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo

#else
    // see ggv-/issues/gui_broken_photon_record_colors.rst note the shift of one to the right of the (j,k,l)

    //                                               j k l  sz   type                  norm   iatt   item_from_dim
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0 ,4,ViewNPY::SHORT          ,true,  false, 2);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,0,1,0 ,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,0,1,2 ,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,0,1,2 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo

#endif

    // structured record array => item_from_dim=2 the count comes from product of 1st two dimensions

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
    m_phosel_data = phosel_data ;
    if(!m_phosel_data) return ; 
    setBufferControl(m_phosel_data);

    //                                               j k l sz   type                norm   iatt   item_from_dim
    ViewNPY* psel = new ViewNPY("psel",m_phosel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true, 1);
    m_phosel_attr = new MultiViewNPY("phosel_attr");
    m_phosel_attr->add(psel);


/*
psel is not currently used in shaders, as have not done much at photon level, only record level

delta:gl blyth$ find . -type f -exec grep -H psel {} \;
delta:gl blyth$ 
*/

}


void OpticksEvent::setRecselData(NPY<unsigned char>* recsel_data)
{
    m_recsel_data = recsel_data ;
    if(!m_recsel_data) return ; 
    setBufferControl(m_recsel_data);
    //                                               j k l sz   type                norm   iatt   item_from_dim
    ViewNPY* rsel = new ViewNPY("rsel",m_recsel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true, 2);
    // structured recsel array, means the count needs to come from product of 1st two dimensions, 

    m_recsel_attr = new MultiViewNPY("recsel_attr");
    m_recsel_attr->add(rsel);


/*

delta:gl blyth$ find . -type f -exec grep -H rsel {} \;
./altrec/vert.glsl:layout(location = 3) in ivec4 rsel;  
./altrec/vert.glsl:    sel = rsel ; 
./devrec/vert.glsl:layout(location = 3) in ivec4 rsel;  
./devrec/vert.glsl:    sel = rsel ; 
./rec/vert.glsl:layout(location = 3) in ivec4 rsel;  
./rec/vert.glsl:    sel = rsel ; 

*/

}

void OpticksEvent::setSeedData(NPY<unsigned>* seed_data)
{
    setBufferControl(seed_data);
    m_seed_data = seed_data  ;
    m_seed_attr = new MultiViewNPY("seed_attr");

}

void OpticksEvent::setSequenceData(NPY<unsigned long long>* sequence_data)
{
    setBufferControl(sequence_data);

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
    //                                                 j k l sz   type                norm   iatt    item_from_dim
    ViewNPY* phis = new ViewNPY("phis",m_sequence_data,0,0,0,4,ViewNPY::UNSIGNED_SHORT,false,  true, 1);
    ViewNPY* pmat = new ViewNPY("pmat",m_sequence_data,0,1,0,4,ViewNPY::UNSIGNED_SHORT,false,  true, 1);
    m_sequence_attr = new MultiViewNPY("sequence_attr");
    m_sequence_attr->add(phis);
    m_sequence_attr->add(pmat);

/*
Looks like the raw photon level sequence data is not used in shaders, instead the rsel (popularity index) 
that is derived from the sequence data by indexing is used::

    delta:gl blyth$ find . -type f -exec grep -H phis {} \;
    delta:gl blyth$ find . -type f -exec grep -H pmat {} \;

*/
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

std::string OpticksEvent::brief()
{
    std::stringstream ss ; 
    ss << "Evt " 
       << getDir()
       << " " << getTimeStamp() 
       << " " << getCreator()
       ;
    return ss.str();
}

std::string OpticksEvent::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " 
       << " id: " << getId()
       << " typ: " << m_typ 
       << " tag: " << m_tag 
       << " det: " << m_det 
       << " cat: " << ( m_cat ? m_cat : "NULL" ) 
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

    NPY<short>* rx = getRecordData() ;
    if(rx && rx->hasData())
        m_parameters->add<std::string>("recordData",   rx->getDigestString()  );

    NPY<unsigned long long>* ph = getSequenceData() ;
    if(ph && ph->hasData())
        m_parameters->add<std::string>("sequenceData", ph->getDigestString()  );
}

void OpticksEvent::save(bool verbose)
{
    bool hpd = hasPhotonData();
    if(!hpd)
    {
        LOG(warning) << "OpticksEvent::save SKIP as no photon data " ; 
        return ; 
    }


    (*m_timer)("_save");


    recordDigests();

    const char* udet = getUDet();
    LOG(info) << description("OpticksEvent::save")
              << getShapeString()
              << " dir " << m_event_spec->getDir() 
              ;    


    NPY<float>* no = getNopstepData();
    if(no)
    {
        no->setVerbose(verbose);
        no->save("no", m_typ,  m_tag, udet);
        no->dump("OpticksEvent::save (nopstep)");
    }
    
   // genstep were formally not saved as they exist already elsewhere,
   // however recording the gs in use for posterity makes sense
    NPY<float>* gs = getGenstepData();
    if(gs)
    {
        gs->setVerbose(verbose);
        gs->save("gs", m_typ,  m_tag, udet);
    }
    else
    {
        LOG(warning) << "failed to getGenstepData" ; 
    }

    NPY<float>* ox = getPhotonData();
    {
        ox->setVerbose(verbose);
        ox->save("ox", m_typ,  m_tag, udet);
    } 

    NPY<short>* rx = getRecordData();    
    {
        rx->setVerbose(verbose);
        rx->save("rx", m_typ,  m_tag, udet);
    }

    NPY<unsigned long long>* ph = getSequenceData();
    {
        ph->setVerbose(verbose);
        ph->save("ph", m_typ,  m_tag, udet);
    }

    NPY<unsigned>* se  = getSeedData();
    {
        se->setVerbose(verbose);
        se->save("se", m_typ,  m_tag, udet);
    }


    updateDomainsBuffer();

    NPY<float>* fdom = getFDomain();
    if(fdom) fdom->save(fdom_, m_typ,  m_tag, udet);

    NPY<int>* idom = getIDomain();
    if(idom) idom->save(idom_, m_typ,  m_tag, udet);

    if(no)
    {
       assert(idom && "OpticksEvent::save non-null nopstep BUT HAS NULL IDOM ");
    }


    bool is_indexed = isIndexed();
    if(is_indexed)
    {
        saveIndex(verbose);
    }
    else
    {
        LOG(warning) << "OpticksEvent::save SKIP saveIndex as not indexed " ; 
    }
   

    saveParameters();

    (*m_timer)("save");

    makeReport(false);  // after timer save, in order to include that in the report
    saveReport();
}



void OpticksEvent::makeReport(bool verbose)
{
    LOG(info) << "OpticksEvent::makeReport " << getTagDir()  ; 

    if(verbose)
    m_parameters->dump();

    m_timer->stop();

    m_ttable = m_timer->makeTable();
    if(verbose)
    m_ttable->dump("OpticksEvent::makeReport");

    m_report->add(m_parameters->getLines());
    m_report->add(m_ttable->getLines());
}


void OpticksEvent::saveReport()
{
    std::string tagdir = getTagDir();
    saveReport(tagdir.c_str());

    std::string anno = getTimeStamp() ;
    std::string tagdir_ts = getTagDir(anno.c_str());
    saveReport(tagdir_ts.c_str());
}



std::string OpticksEvent::TagDir(const char* det, const char* typ, const char* tag, const char* anno)
{
    std::string tagdir = BOpticksEvent::directory(det, typ, tag, anno ? anno : NULL );
    return tagdir ; 

}
std::string OpticksEvent::getTagDir(const char* anno)
{
    const char* udet = getUDet();
    std::string tagdir = TagDir(udet, m_typ, m_tag, anno ? anno : NULL );
    return tagdir ;
}


void OpticksEvent::saveParameters()
{
    std::string tagdir = getTagDir();
    m_parameters->save(tagdir.c_str(), PARAMETERS_NAME);

    std::string anno = getTimeStamp() ;
    std::string tagdir_ts = getTagDir(anno.c_str());
    m_parameters->save(tagdir_ts.c_str(), PARAMETERS_NAME);
}


void OpticksEvent::loadParameters()
{
    std::string tagdir = getTagDir();
    m_parameters->load_(tagdir.c_str(), PARAMETERS_NAME );
}

void OpticksEvent::importParameters()
{
    std::string mode_ = m_parameters->get<std::string>("mode"); 
    OpticksMode* mode = new OpticksMode(mode_.c_str());
    LOG(info) << "OpticksEvent::importParameters "
              << " mode_ " << mode_ 
              << " --> " << mode->description() ; 
    setMode(mode);
}



void OpticksEvent::saveReport(const char* dir)
{
    if(!m_ttable || !m_report) return ; 
    LOG(debug) << "OpticksEvent::saveReport to " << dir  ; 

    m_ttable->save(dir);
    m_report->save(dir);  
}

void OpticksEvent::loadReport()
{
    std::string tagdir = getTagDir();
    m_ttable = Timer::loadTable(tagdir.c_str());
    m_report = Report::load(tagdir.c_str());
}

void OpticksEvent::setFakeNopstepPath(const char* path)
{
    // fake path used by OpticksEvent::load rather than standard one
    // see npy-/nopstep_viz_debug.py

    m_fake_nopstep_path = path ? strdup(path) : NULL ;
}


OpticksEvent* OpticksEvent::load(const char* typ, const char* tag, const char* det, const char* cat, bool verbose)
{
    LOG(info) << "OpticksEvent::load"
              << " typ " << typ
              << " tag " << tag
              << " det " << det
              << " cat " << ( cat ? cat : "NULL" )
              ;


    OpticksEventSpec* spec = new OpticksEventSpec(typ, tag, det, cat);
    OpticksEvent* evt = new OpticksEvent(spec);

    evt->loadBuffers(verbose);

    if(evt->isNoLoad())
    {
         LOG(warning) << "OpticksEvent::load FAILED " ;
         delete evt ;
         evt = NULL ;
    } 

    return evt ;  
}



void OpticksEvent::loadBuffersImportSpec(NPYBase* npy, NPYSpec* spec)
{
    assert(npy->hasItemSpec(spec));
    npy->setBufferSpec(spec);
}


const char* OpticksEvent::getPath(const char* xx)
{
    std::string name = m_abbrev.count(xx) == 1 ? m_abbrev[xx] : xx ;  
    const char* udet = getUDet(); // cat overrides det if present 
    std::string path = BOpticksEvent::path(udet, m_typ, m_tag, name.c_str() );
    return strdup(path.c_str()) ; 
}


void OpticksEvent::loadBuffers(bool verbose)
{
    TIMER("_load");

    const char* udet = getUDet(); // cat overrides det if present 

    bool qload = true ; 

    NPY<int>*   idom = NPY<int>::load(idom_, m_typ,  m_tag, udet, qload);

    if(!idom)
    {
        std::string tagdir = getTagDir();

        m_noload = true ; 
        LOG(warning) << "OpticksEvent::load NO SUCH EVENT : RUN WITHOUT --load OPTION TO CREATE IT " 
                     << " typ: " << m_typ
                     << " tag: " << m_tag
                     << " det: " << m_det
                     << " cat: " << ( m_cat ? m_cat : "NULL" )
                     << " udet: " << udet 
                     << " tagdir " << tagdir    
                    ;     
        return ; 
    }

    m_loaded = true ; 

    NPY<float>* fdom = NPY<float>::load(fdom_, m_typ,  m_tag, udet, qload );

    setIDomain(idom);
    setFDomain(fdom);

    loadReport();
    loadParameters();

    importParameters();
    loadIndex();

    importDomainsBuffer();

    createSpec();      // domains import sets maxrec allowing spec to be created 

    assert(idom->hasShapeSpec(m_idom_spec));
    assert(fdom->hasShapeSpec(m_fdom_spec));
 
    NPY<float>* no = NULL ; 
    if(m_fake_nopstep_path)
    {
        LOG(warning) << "OpticksEvent::loadBuffers using setFakeNopstepPath " << m_fake_nopstep_path ; 
        no = NPY<float>::debugload(m_fake_nopstep_path);
    }
    else
    {  
        no = NPY<float>::load("no", m_typ,  m_tag, udet, qload);
    }
    if(no) loadBuffersImportSpec(no, m_nopstep_spec) ;

    NPY<float>*              gs = NPY<float>::load("gs", m_typ,  m_tag, udet, qload);
    NPY<float>*              ox = NPY<float>::load("ox", m_typ,  m_tag, udet, qload);
    NPY<short>*              rx = NPY<short>::load("rx", m_typ,  m_tag, udet, qload);

    NPY<unsigned long long>* ph = NPY<unsigned long long>::load("ph", m_typ,  m_tag, udet, qload );
    NPY<unsigned char>*      ps = NPY<unsigned char>::load("ps", m_typ,  m_tag, udet, qload );
    NPY<unsigned char>*      rs = NPY<unsigned char>::load("rs", m_typ,  m_tag, udet, qload );
    NPY<unsigned>*           se = NPY<unsigned>::load("se", m_typ,  m_tag, udet, qload );

    if(ph == NULL || ps == NULL || rs == NULL )
        LOG(warning) << "OpticksEvent::loadBuffers " << getDir()
                     << " MISSING INDEX BUFFER(S) " 
                     << " ph " << ph
                     << " ps " << ps
                     << " rs " << rs
                     ;


    if(gs) loadBuffersImportSpec(gs,m_genstep_spec) ;
    if(ox) loadBuffersImportSpec(ox,m_photon_spec) ;
    if(rx) loadBuffersImportSpec(rx,m_record_spec) ;
    if(ph) loadBuffersImportSpec(ph,m_sequence_spec) ;
    if(ps) loadBuffersImportSpec(ps,m_phosel_spec) ;
    if(rs) loadBuffersImportSpec(rs,m_recsel_spec) ;
    if(se) loadBuffersImportSpec(se,m_seed_spec) ;


    if(gs) importGenstepDataLoaded(gs);   // sets action control, so setGenstepData label checks can succeed

    unsigned int num_genstep = gs ? gs->getShape(0) : 0 ;
    unsigned int num_nopstep = no ? no->getShape(0) : 0 ;
    unsigned int num_photons = ox ? ox->getShape(0) : 0 ;
    unsigned int num_history = ph ? ph->getShape(0) : 0 ;
    unsigned int num_phosel  = ps ? ps->getShape(0) : 0 ;
    unsigned int num_seed    = se ? se->getShape(0) : 0 ;

    // either zero or matching 
    assert(num_history == 0 || num_photons == num_history );
    assert(num_phosel == 0 || num_photons == num_phosel );
    assert(num_seed == 0 || num_photons == num_seed );

    unsigned int num_records = rx ? rx->getShape(0) : 0 ;
    unsigned int num_recsel  = rs ? rs->getShape(0) : 0 ;

    assert(num_recsel == 0 || num_records == num_recsel );


    LOG(info) << "OpticksEvent::load shape(0) before reshaping "
              << " num_genstep " << num_genstep
              << " num_nopstep " << num_nopstep
              << " [ "
              << " num_photons " << num_photons
              << " num_history " << num_history
              << " num_phosel " << num_phosel 
              << " num_seed " << num_seed 
              << " ] "
              << " [ "
              << " num_records " << num_records
              << " num_recsel " << num_recsel
              << " ] "
              ; 


    // treat "persisted for posterity" gensteps just like all other buffers
    // progenitor input gensteps need different treatment

    bool progenitor = false; 
    setGenstepData(gs, progenitor);
    setNopstepData(no);
    setPhotonData(ox);
    setSequenceData(ph);
    setRecordData(rx);

    setPhoselData(ps);
    setRecselData(rs);
    setSeedData(se);

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
        if(ps) ps->Summary("ps");
        if(rs) rs->Summary("rs");
        if(se) se->Summary("se");
    }

    if(!isIndexed())
    {
         LOG(warning) << "OpticksEvent::load IS NOT INDEXED " 
                      << brief()
                      ;
    }




}

bool OpticksEvent::isIndexed()
{
    return m_phosel_data != NULL && m_recsel_data != NULL && m_seqhis != NULL && m_seqmat != NULL ;
}


NPY<float>* OpticksEvent::loadGenstepDerivativeFromFile(const char* stem)
{
    std::string path = BOpticksEvent::path(m_det, m_typ, m_tag, stem, ".npy" ); // top/sub/tag/stem/ext
    bool exists = BFile::ExistsFile(path.c_str()) ;

    if(exists)
    LOG(info) << "OpticksEvent::loadGenstepDerivativeFromFile  "
              << " m_det " << m_det
              << " m_typ " << m_typ
              << " m_tag " << m_tag
              << " stem " << stem
              << " --> " << path 
              ;
 
    NPY<float>* npy = exists ? NPY<float>::load(path.c_str()) : NULL ;
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
   return m_parameters->get<int>("NumG4Event","1");
}
unsigned int OpticksEvent::getNumPhotonsPerG4Event()
{
   return m_parameters->get<int>("NumPhotonsPerG4Event","0");  // "0" : fallback if not set (eg for G4GUN running )
}
 
void OpticksEvent::postPropagateGeant4()
{
    unsigned int num_photons = m_photon_data->getShape(0);
    LOG(info) << "OpticksEvent::postPropagateGeant4"
              << " shape " << getShapeString()
              << " num_photons " << num_photons
              ;

    setNumPhotons(num_photons);  
   // triggers resize ???  THIS IS ONLY NEED FOR DYNAMIC RUNNING 
   // WITH FABRICATED OR LOADED GENSTEPS THIS IS KNOWN AHEAD OF TIME

    indexPhotonsCPU();    
}

void OpticksEvent::indexPhotonsCPU()
{
    // see tests/IndexerTest

    OK_PROFILE("_OpticksEvent::indexPhotonsCPU");

    NPY<unsigned long long>* sequence = getSequenceData();
    NPY<unsigned char>*        phosel = getPhoselData();
    NPY<unsigned char>*        recsel0 = getRecselData();

    LOG(info) << "OpticksEvent::indexPhotonsCPU" 
              << " sequence " << sequence->getShapeString()
              << " phosel "   << phosel->getShapeString()
              << " phosel.hasData "   << phosel->hasData()
              << " recsel0 "   << recsel0->getShapeString()
              << " recsel0.hasData "   << recsel0->hasData()
              ;

    unsigned int maxrec = getMaxRec();

    assert(sequence->hasItemShape(1,2));
    assert(phosel->hasItemShape(1,4));
    assert(recsel0->hasItemShape(maxrec,1,4));
    assert(sequence->getShape(0) == phosel->getShape(0));
    assert(sequence->getShape(0) == recsel0->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence(OpticksConst::SEQHIS_NAME_, OpticksConst::SEQMAT_NAME_);

    assert(!phosel->hasData()) ; 

    phosel->zero();
    unsigned char* phosel_values = phosel->getValues() ;
    assert(phosel_values);
    idx->applyLookup<unsigned char>(phosel_values);



    NPY<unsigned char>* recsel1 = NPY<unsigned char>::make_repeat(phosel, maxrec ) ;
    recsel1->reshape(-1, maxrec, 1, 4);
    recsel1->setBufferSpec(m_recsel_spec);  

    if(recsel0 && recsel0->hasData()) LOG(warning) << " leaking recsel0 " ; 

    setRecselData(recsel1);


    setHistoryIndex(idx->getHistoryIndex());
    setMaterialIndex(idx->getMaterialIndex());

    TIMER("indexPhotonsCPU");    

    OK_PROFILE("OpticksEvent::indexPhotonsCPU");
}




void OpticksEvent::saveIndex(bool verbose_)
{


    const char* udet = getUDet();

    NPYBase::setGlobalVerbose(verbose_);

    NPY<unsigned char>* ps = getPhoselData();
    NPY<unsigned char>* rs = getRecselData();

    assert(ps);
    assert(rs);

    ps->save("ps", m_typ,  m_tag, udet);
    rs->save("rs", m_typ,  m_tag, udet);

    NPYBase::setGlobalVerbose(false);

    std::string tagdir = getTagDir();
    LOG(info) << "OpticksEvent::saveIndex"
              << " tagdir " << tagdir
              << " seqhis " << m_seqhis
              << " seqmat " << m_seqmat
              << " bndidx " << m_bndidx
              ; 

    if(m_seqhis)
        m_seqhis->save(tagdir.c_str());        
    else
        LOG(warning) << "OpticksEvent::saveIndex no seqhis to save " ;

    if(m_seqmat)
        m_seqmat->save(tagdir.c_str());        
    else
        LOG(warning) << "OpticksEvent::saveIndex no seqmat to save " ;

    if(m_bndidx)
        m_bndidx->save(tagdir.c_str());        
    else
        LOG(warning) << "OpticksEvent::saveIndex no bndidx to save " ;
}

void OpticksEvent::loadIndex()
{
    std::string tagdir_ = getTagDir();
    const char* tagdir = tagdir_.c_str();

    m_seqhis = Index::load(tagdir, OpticksConst::SEQHIS_NAME_ );
    m_seqmat = Index::load(tagdir, OpticksConst::SEQMAT_NAME_ );  
    m_bndidx = Index::load(tagdir, OpticksConst::BNDIDX_NAME_ );

    LOG(debug) << "OpticksEvent::loadIndex"
              << " tagdir " << tagdir 
              << " seqhis " << m_seqhis 
              << " seqmat " << m_seqmat 
              << " bndidx " << m_bndidx 
              ;
}



Index* OpticksEvent::loadNamedIndex( const char* typ, const char* tag, const char* udet, const char* name)
{
    //const char* species = "ix" ; 
    //std::string ixdir = speciesDir(species, udet, typ);
    std::string tagdir = TagDir(udet, typ, tag);
    Index* seqhis = Index::load(tagdir.c_str(), name );
    return seqhis ; 
}

Index* OpticksEvent::loadHistoryIndex( const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(typ, tag, udet, OpticksConst::SEQHIS_NAME_); 
}
Index* OpticksEvent::loadMaterialIndex( const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(typ, tag, udet, OpticksConst::SEQMAT_NAME_); 
}
Index* OpticksEvent::loadBoundaryIndex( const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(typ, tag, udet, OpticksConst::BNDIDX_NAME_); 
}




