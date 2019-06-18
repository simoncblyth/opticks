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


// brap-
#include "BTimes.hh"
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
#include "NGeoTestConfig.hpp"
#include "NMeta.hpp"

#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "TrivialCheckNPY.hpp"
#include "GLMFormat.hpp"
#include "Index.hpp"

#include "Report.hpp"
#include "BTimeKeeper.hh"
#include "BTimes.hh"
#include "BTimesTable.hh"


#include "OKConf.hh"

// okc-
#include "Opticks.hh"
#include "OpticksProfile.hh"
#include "OpticksSwitches.h"
#include "OpticksPhoton.h"
#include "OpticksConst.hh"
#include "OpticksDomain.hh"
#include "OpticksFlags.hh"
#include "OpticksEventInstrument.hh"
#include "OpticksEvent.hh"
#include "OpticksMode.hh"
#include "OpticksBufferSpec.hh"
#include "OpticksBufferControl.hh"
#include "OpticksActionControl.hh"
#include "Indexer.hh"

#include "PLOG.hh"


const plog::Severity OpticksEvent::LEVEL = debug ; 


const char* OpticksEvent::TIMEFORMAT = "%Y%m%d_%H%M%S" ;
const char* OpticksEvent::PARAMETERS_NAME = "parameters.json" ;
const char* OpticksEvent::PARAMETERS_STEM = "parameters" ;
const char* OpticksEvent::PARAMETERS_EXT = ".json" ;


std::string OpticksEvent::timestamp()
{
    std::string timestamp = BTime::now(TIMEFORMAT, 0);
    return timestamp ; 
}


bool OpticksEvent::CanAnalyse(OpticksEvent* evt)
{
    return evt && evt->hasRecordData() ; 
}


const char* OpticksEvent::fdom_    = "fdom" ; 
const char* OpticksEvent::idom_    = "idom" ; 
const char* OpticksEvent::genstep_ = "genstep" ; 
const char* OpticksEvent::nopstep_ = "nopstep" ; 
const char* OpticksEvent::photon_  = "photon" ; 
const char* OpticksEvent::source_  = "source" ; 
const char* OpticksEvent::record_  = "record" ; 
const char* OpticksEvent::phosel_ = "phosel" ; 
const char* OpticksEvent::recsel_  = "recsel" ; 
const char* OpticksEvent::sequence_  = "sequence" ; 
const char* OpticksEvent::seed_  = "seed" ; 
const char* OpticksEvent::hit_  = "hit" ; 


OpticksEvent* OpticksEvent::make(OpticksEventSpec* spec, unsigned tagoffset)
{
     OpticksEventSpec* offspec = spec->clone(tagoffset);
     return new OpticksEvent(offspec) ; 
}

Opticks* OpticksEvent::getOpticks() const { return m_ok ; }
OpticksProfile* OpticksEvent::getProfile() const { return m_profile ; }


const char* OpticksEvent::PRELAUNCH_LABEL = "OpticksEvent_prelaunch" ;
const char* OpticksEvent::LAUNCH_LABEL = "OpticksEvent_launch" ; 
 

OpticksEvent::OpticksEvent(OpticksEventSpec* spec) 
          :
          OpticksEventSpec(spec),
          m_event_spec(spec),
          m_ok(NULL),   // set by Opticks::makeEvent
          m_profile(NULL),

          m_noload(false),
          m_loaded(false),
          m_versions(NULL),
          m_parameters(NULL),
          m_report(NULL),

          m_primary_data(NULL),
          m_genstep_data(NULL),
          m_nopstep_data(NULL),
          m_photon_data(NULL),
          m_source_data(NULL),
          m_record_data(NULL),
          m_phosel_data(NULL),
          m_recsel_data(NULL),
          m_sequence_data(NULL),
          m_seed_data(NULL),
          m_hit_data(NULL),

          m_photon_ctrl(NULL),
          m_source_ctrl(NULL),
          m_seed_ctrl(NULL),
          m_domain(NULL),

          m_genstep_vpos(NULL),
          m_genstep_attr(NULL),
          m_nopstep_attr(NULL),
          m_photon_attr(NULL),
          m_source_attr(NULL),
          m_record_attr(NULL),
          m_phosel_attr(NULL),
          m_recsel_attr(NULL),
          m_sequence_attr(NULL),
          m_seed_attr(NULL),
          m_hit_attr(NULL),

          m_records(NULL),
          m_photons(NULL),
          m_hits(NULL),
          m_bnd(NULL),

          m_num_gensteps(0),
          m_num_nopsteps(0),
          m_num_photons(0),
          m_num_source(0),

          m_seqhis(NULL),
          m_seqmat(NULL),
          m_bndidx(NULL),
          m_fake_nopstep_path(NULL),

          m_fdom_spec(NULL),
          m_idom_spec(NULL),
          m_genstep_spec(NULL),
          m_nopstep_spec(NULL),
          m_photon_spec(NULL),
          m_source_spec(NULL),
          m_record_spec(NULL),
          m_phosel_spec(NULL),
          m_recsel_spec(NULL),
          m_sequence_spec(NULL),

          m_prelaunch_times(new BTimes(PRELAUNCH_LABEL)),
          m_launch_times(new BTimes(LAUNCH_LABEL)),
          m_sibling(NULL),
          m_geopath(NULL),
          m_geotestconfig(NULL)

{
    init();
}


OpticksEvent::~OpticksEvent()
{
    LOG(info) << "OpticksEvent::~OpticksEvent PLACEHOLDER" ; 
} 


BTimes* OpticksEvent::getPrelaunchTimes()
{
    return m_prelaunch_times ; 
}
BTimes* OpticksEvent::getLaunchTimes()
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





bool OpticksEvent::isNoLoad() const 
{
    return m_noload ; 
}
bool OpticksEvent::isLoaded() const 
{
    return m_loaded ; 
}
bool OpticksEvent::isStep() const 
{ 
    return true  ; 
}
bool OpticksEvent::isFlat() const 
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
    bool resize_ = true ; 
    setNumPhotons(0, resize_);
}

void OpticksEvent::setNumPhotons(unsigned int num_photons, bool resize_)  // resize_ default true 
{
    m_num_photons = num_photons ; 
    if(resize_)
    {
        LOG(verbose) << "OpticksEvent::setNumPhotons RESIZING " << num_photons ;  
        resize();
    }
    else
    {
        LOG(verbose) << "OpticksEvent::setNumPhotons NOT RESIZING " << num_photons ;  
    }
}
unsigned int OpticksEvent::getNumPhotons() const 
{
    return m_num_photons ; 
}
unsigned int OpticksEvent::getNumSource() const 
{
    return m_num_source ; 
}





unsigned int OpticksEvent::getNumRecords() const 
{
    unsigned int maxrec = getMaxRec();
    return m_num_photons * maxrec ; 
}




bool OpticksEvent::hasGenstepData() const
{
    return m_genstep_data && m_genstep_data->hasData() ; 
}
bool OpticksEvent::hasSourceData() const 
{
    return m_source_data && m_source_data->hasData() ; 
}
bool OpticksEvent::hasPhotonData() const 
{
    return m_photon_data && m_photon_data->hasData() ; 
}
bool OpticksEvent::hasRecordData() const
{
    return m_record_data && m_record_data->hasData() ; 
}




NPY<float>* OpticksEvent::getGenstepData() const 
{ 
     return m_genstep_data ;
}
NPY<float>* OpticksEvent::getNopstepData() const  
{ 
     return m_nopstep_data ; 
}
NPY<float>* OpticksEvent::getPhotonData() const 
{
     return m_photon_data ; 
} 
NPY<float>* OpticksEvent::getSourceData() const 
{
     return m_source_data ; 
} 
NPY<short>* OpticksEvent::getRecordData() const  
{ 
    return m_record_data ; 
}
NPY<unsigned char>* OpticksEvent::getPhoselData() const 
{ 
    return m_phosel_data ;
}
NPY<unsigned char>* OpticksEvent::getRecselData() const 
{ 
    return m_recsel_data ; 
}
NPY<unsigned long long>* OpticksEvent::getSequenceData() const 
{ 
    return m_sequence_data ;
}
NPY<unsigned>* OpticksEvent::getSeedData() const 
{ 
    return m_seed_data ;
}
NPY<float>* OpticksEvent::getHitData() const 
{ 
    return m_hit_data ;
}

MultiViewNPY* OpticksEvent::getGenstepAttr(){ return m_genstep_attr ; }
MultiViewNPY* OpticksEvent::getNopstepAttr(){ return m_nopstep_attr ; }
MultiViewNPY* OpticksEvent::getPhotonAttr(){ return m_photon_attr ; }
MultiViewNPY* OpticksEvent::getSourceAttr(){ return m_source_attr ; }
MultiViewNPY* OpticksEvent::getRecordAttr(){ return m_record_attr ; }
MultiViewNPY* OpticksEvent::getPhoselAttr(){ return m_phosel_attr ; }
MultiViewNPY* OpticksEvent::getRecselAttr(){ return m_recsel_attr ; }
MultiViewNPY* OpticksEvent::getSequenceAttr(){ return m_sequence_attr ; }
MultiViewNPY* OpticksEvent::getSeedAttr(){   return m_seed_attr ; }
MultiViewNPY* OpticksEvent::getHitAttr(){    return m_hit_attr ; }






void OpticksEvent::setRecordsNPY(RecordsNPY* records)
{
    m_records = records ; 
}
RecordsNPY* OpticksEvent::getRecordsNPY()
{
    if(m_records == NULL)
    {
        m_records = OpticksEventInstrument::CreateRecordsNPY(this) ;

        if(!m_records)
            LOG(error) << "failed to CreateRecordsNPY " 
            ; 
        //assert( m_records ); 
    }
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





//////////////// m_domain related ///////////////////////////////////

void OpticksEvent::setFDomain(NPY<float>* fdom) { m_domain->setFDomain(fdom) ; } 
void OpticksEvent::setIDomain(NPY<int>* idom) {   m_domain->setIDomain(idom) ; } 

NPY<float>* OpticksEvent::getFDomain() const { return m_domain->getFDomain() ; } 
NPY<int>*   OpticksEvent::getIDomain() const { return m_domain->getIDomain() ; }

// below set by Opticks::makeEvent

void OpticksEvent::setSpaceDomain(const glm::vec4& space_domain) {           m_domain->setSpaceDomain(space_domain) ; } 
void OpticksEvent::setTimeDomain(const glm::vec4& time_domain) {             m_domain->setTimeDomain(time_domain)  ; } 
void OpticksEvent::setWavelengthDomain(const glm::vec4& wavelength_domain) { m_domain->setWavelengthDomain(wavelength_domain)  ; } 

const glm::vec4& OpticksEvent::getSpaceDomain() const {      return m_domain->getSpaceDomain() ; } 
const glm::vec4& OpticksEvent::getTimeDomain() const {       return m_domain->getTimeDomain() ; } 
const glm::vec4& OpticksEvent::getWavelengthDomain() const { return m_domain->getWavelengthDomain() ; } 

unsigned OpticksEvent::getMaxRec() const {     return m_domain->getMaxRec() ; } 
unsigned OpticksEvent::getMaxBounce() const  { return m_domain->getMaxBounce() ; }  // caution there is a getBounceMax 
unsigned OpticksEvent::getMaxRng() const {     return m_domain->getMaxRng() ; } 

void OpticksEvent::setMaxRec(unsigned maxrec) {       m_domain->setMaxRec(maxrec); } 
void OpticksEvent::setMaxBounce(unsigned maxbounce) { m_domain->setMaxBounce(maxbounce); } 
void OpticksEvent::setMaxRng(unsigned maxrng) {       m_domain->setMaxRng(maxrng); } 

void OpticksEvent::dumpDomains(const char* msg)
{
    m_domain->dump(msg);
}
void OpticksEvent::updateDomainsBuffer()
{
    m_domain->updateBuffer();
}
void OpticksEvent::importDomainsBuffer()  // invoked by OpticksEvent::loadBuffers
{
    m_domain->importBuffer();
}
void OpticksEvent::saveDomains()   // invoked by OpticksEvent::save
{
    updateDomainsBuffer();

    NPY<float>* fdom = getFDomain();
    if(fdom) fdom->save(m_pfx, fdom_, m_typ,  m_tag, m_udet);

    NPY<int>* idom = getIDomain();
    if(idom) idom->save(m_pfx, idom_, m_typ,  m_tag, m_udet);
}

////////////////////////////////////////////////////////////////////////////////////





void OpticksEvent::setMode(OpticksMode* mode)
{ 
    m_mode = mode ; 
}
bool OpticksEvent::isInterop()
{
    return m_mode->isInterop();
}
bool OpticksEvent::isCompute()
{
    return m_mode->isCompute();
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


NMeta*      OpticksEvent::getParameters()
{
    return m_parameters ;
}


void OpticksEvent::pushNames(std::vector<std::string>& names)
{
    names.push_back(genstep_);
    names.push_back(nopstep_);
    names.push_back(photon_);
    names.push_back(source_);
    names.push_back(record_);
    names.push_back(phosel_);
    names.push_back(recsel_);
    names.push_back(sequence_);
    names.push_back(seed_);
    names.push_back(hit_);
} 

/**
OpticksEvent::init
-------------------

From ipython ab.py testing::

    In [1]: a.metadata.parameters
    Out[1]: 
    {u'BounceMax': 9,
     u'Cat': u'tboolean-box',
     u'Creator': u'/home/blyth/local/opticks/lib/OKG4Test',
     u'Detector': u'tboolean-box',
     u'EntryCode': u'G',
     u'EntryName': u'GENERATE',
     u'GEOCACHE': u'/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1',
     u'Id': 1,
     u'KEY': u'OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce',
     u'NumGensteps': 1,


**/




void OpticksEvent::init()
{
    m_versions = new NMeta ;
    m_parameters = new NMeta ;
    m_report = new Report ; 
    m_domain = new OpticksDomain ; 

    m_versions->add<int>("OptiXVersion",  OKConf::OptiXVersionInteger() );
    m_versions->add<int>("CUDAVersion",   OKConf::CUDAVersionInteger() );
    m_versions->add<int>("ComputeVersion", OKConf::ComputeCapabilityInteger() );
    m_versions->add<int>("Geant4Version",  OKConf::Geant4VersionInteger() );

    m_parameters->add<std::string>("TimeStamp", timestamp() );
    m_parameters->add<std::string>("Type", m_typ );
    m_parameters->add<std::string>("Tag", m_tag );
    m_parameters->add<std::string>("Detector", m_det );
    if(m_cat) m_parameters->add<std::string>("Cat", m_cat );
    m_parameters->add<std::string>("UDet", getUDet() );

    std::string switches = OpticksSwitches(); 
    m_parameters->add<std::string>("Switches", switches );

    pushNames(m_data_names);

    m_abbrev[genstep_] = "gs" ;    // input gs are named: cerenkov, scintillation but for posterity need common output tag
    m_abbrev[nopstep_] = "no" ;    // non optical particle steps obtained from G4 eg with g4gun
    m_abbrev[photon_] = "ox" ;     // photon final step uncompressed 
    m_abbrev[source_] = "so" ;     // input photon  
    m_abbrev[record_] = "rx" ;     // photon step compressed record
    m_abbrev[phosel_] = "ps" ;     // photon selection index
    m_abbrev[recsel_] = "rs" ;     // record selection index
    m_abbrev[sequence_] = "ph" ;   // (unsigned long long) photon seqhis/seqmat
    m_abbrev[seed_] = "se" ;   //   (short) genstep id used for photon seeding 
    m_abbrev[hit_] = "ht" ;  
}




NPYBase* OpticksEvent::getData(const char* name)
{
    NPYBase* data = NULL ; 
    if(     strcmp(name, genstep_)==0) data = static_cast<NPYBase*>(m_genstep_data) ; 
    else if(strcmp(name, nopstep_)==0) data = static_cast<NPYBase*>(m_nopstep_data) ;
    else if(strcmp(name, photon_)==0)  data = static_cast<NPYBase*>(m_photon_data) ;
    else if(strcmp(name, source_)==0)  data = static_cast<NPYBase*>(m_source_data) ;
    else if(strcmp(name, record_)==0)  data = static_cast<NPYBase*>(m_record_data) ;
    else if(strcmp(name, phosel_)==0)  data = static_cast<NPYBase*>(m_phosel_data) ;
    else if(strcmp(name, recsel_)==0)  data = static_cast<NPYBase*>(m_recsel_data) ;
    else if(strcmp(name, sequence_)==0) data = static_cast<NPYBase*>(m_sequence_data) ;
    else if(strcmp(name, seed_)==0) data = static_cast<NPYBase*>(m_seed_data) ;
    else if(strcmp(name, hit_)==0) data = static_cast<NPYBase*>(m_hit_data) ;
    return data ; 
}

NPYSpec* OpticksEvent::getSpec(const char* name)
{
    NPYSpec* spec = NULL ; 
    if(     strcmp(name, genstep_)==0) spec = static_cast<NPYSpec*>(m_genstep_spec) ; 
    else if(strcmp(name, nopstep_)==0) spec = static_cast<NPYSpec*>(m_nopstep_spec) ;
    else if(strcmp(name, photon_)==0)  spec = static_cast<NPYSpec*>(m_photon_spec) ;
    else if(strcmp(name, source_)==0)  spec = static_cast<NPYSpec*>(m_source_spec) ;
    else if(strcmp(name, record_)==0)  spec = static_cast<NPYSpec*>(m_record_spec) ;
    else if(strcmp(name, phosel_)==0)  spec = static_cast<NPYSpec*>(m_phosel_spec) ;
    else if(strcmp(name, recsel_)==0)  spec = static_cast<NPYSpec*>(m_recsel_spec) ;
    else if(strcmp(name, sequence_)==0) spec = static_cast<NPYSpec*>(m_sequence_spec) ;
    else if(strcmp(name, seed_)==0)     spec = static_cast<NPYSpec*>(m_seed_spec) ;
    else if(strcmp(name, hit_)==0)     spec = static_cast<NPYSpec*>(m_hit_spec) ;
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
    m_profile = ok->getProfile(); 
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


void OpticksEvent::setTestCSGPath(const char* testcsgpath)
{
    m_parameters->add<std::string>("TestCSGPath", testcsgpath ? testcsgpath : "" );
}
std::string OpticksEvent::getTestCSGPath()
{
    return m_parameters->get<std::string>("TestCSGPath", "");
}

void OpticksEvent::setTestConfigString(const char* testconfig)
{
    m_parameters->add<std::string>("TestConfig", testconfig ? testconfig : "" );
}
std::string OpticksEvent::getTestConfigString()
{
    return m_parameters->get<std::string>("TestConfig", "");
}

NGeoTestConfig* OpticksEvent::getTestConfig()
{

    if(m_geotestconfig == NULL)
    {
         std::string gtc = getTestConfigString() ; 
         LOG(info) << " gtc " << gtc ; 
         m_geotestconfig = gtc.empty() ? NULL : new NGeoTestConfig( gtc.c_str() );
    }
    return m_geotestconfig ; 
}






void OpticksEvent::setNote(const char* note)
{
    m_parameters->add<std::string>("Note", note ? note : "NULL" );
}
void OpticksEvent::appendNote(const char* note)
{

#ifdef OLD_PARAMETERS
    m_parameters->append<std::string>("Note", note ? note : "" );
#else
    std::string n = note ? note : "" ; 
    m_parameters->appendString("Note", n );
#endif
}


std::string OpticksEvent::getNote()
{
    return m_parameters->get<std::string>("Note", "");
}





const char* OpticksEvent::getGeoPath()
{
    if(m_geopath == NULL)
    {
        // only test geopath for now 
        std::string testcsgpath_ = getTestCSGPath();

        const char* testcsgpath = testcsgpath_.empty() ? NULL : testcsgpath_.c_str() ;
        const char* dbgcsgpath = m_ok->getDbgCSGPath();
        const char* geopath = testcsgpath ? testcsgpath : ( dbgcsgpath ? dbgcsgpath : NULL ) ; 

        if( testcsgpath && dbgcsgpath && strcmp(testcsgpath, dbgcsgpath) != 0)
        {

            LOG(warning) << "OpticksEvent::getGeoPath"
                         << " BOTH testcsgpath and dbgcsgpath DEFINED AND DIFFERENT "
                         << " testcsgpath " << testcsgpath
                         << " dbgcsgpath " <<  dbgcsgpath
                         << " geopath " <<  geopath
                         ;
        }
 
        m_geopath = geopath ? strdup(geopath) : NULL ; 


        if( geopath == NULL )
        {
            LOG(warning) << "OpticksEvent::getGeoPath"
                         << " FAILED TO RESOLVE GeoPath "
                         << " WORKAROUND EG USE --dbgcsgpath "
                          ; 
        }

    }
    return m_geopath ; 
}





void OpticksEvent::setEntryCode(char entryCode)
{
    m_parameters->add<char>("EntryCode", entryCode );
}
char OpticksEvent::getEntryCode()
{
    return m_parameters->get<char>("EntryCode", "0");
}


void OpticksEvent::setDynamic(int dynamic)
{
    m_parameters->add<int>("Dynamic", dynamic );
}
int OpticksEvent::getDynamic() const 
{
    return m_parameters->get<int>("Dynamic", "-1");
}






void OpticksEvent::setTimeStamp(const char* tstamp)
{
    m_parameters->set<std::string>("TimeStamp", tstamp);
}
std::string OpticksEvent::getTimeStamp()
{
    return m_parameters->get<std::string>("TimeStamp");
}
unsigned int OpticksEvent::getBounceMax() const 
{
    return m_parameters->get<unsigned int>("BounceMax");
}
unsigned int OpticksEvent::getRngMax() const 
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
    else if(elem[0] == source_)   mvn = m_source_attr ;
    else if(elem[0] == record_)   mvn = m_record_attr ;
    else if(elem[0] == phosel_)   mvn = m_phosel_attr ;
    else if(elem[0] == recsel_)   mvn = m_recsel_attr ;
    else if(elem[0] == sequence_) mvn = m_sequence_attr ;
    else if(elem[0] == seed_)     mvn = m_seed_attr ;
    else if(elem[0] == hit_)      mvn = m_hit_attr ;

    assert(mvn);
    return (*mvn)[elem[1].c_str()] ;
}



NPYSpec* OpticksEvent::GenstepSpec(bool compute)
{
    return new NPYSpec(genstep_   ,  0,6,4,0,0,      NPYBase::FLOAT     , OpticksBufferSpec::Get(genstep_, compute))  ;
}
NPYSpec* OpticksEvent::SeedSpec(bool compute)
{
    return new NPYSpec(seed_     ,  0,1,1,0,0,      NPYBase::UINT      , OpticksBufferSpec::Get(seed_, compute)) ;
}
NPYSpec* OpticksEvent::SourceSpec(bool compute)
{
    return new NPYSpec(source_   ,  0,4,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(source_, compute)) ;
}



void OpticksEvent::createSpec()
{
    // invoked by Opticks::makeEvent   or OpticksEvent::load
    unsigned int maxrec = getMaxRec();
    bool compute = isCompute();

    m_genstep_spec = GenstepSpec(compute);
    m_seed_spec    = SeedSpec(compute);
    m_source_spec  = SourceSpec(compute);

    m_hit_spec      = new NPYSpec(hit_       , 0,4,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(hit_, compute));
    m_photon_spec   = new NPYSpec(photon_   ,  0,4,4,0,0,      NPYBase::FLOAT     ,  OpticksBufferSpec::Get(photon_, compute)) ;
    m_record_spec   = new NPYSpec(record_   ,  0,maxrec,2,4,0, NPYBase::SHORT     ,  OpticksBufferSpec::Get(record_, compute)) ;
    //   SHORT -> RT_FORMAT_SHORT4 and size set to  num_quads = num_photons*maxrec*2  

    m_sequence_spec = new NPYSpec(sequence_ ,  0,1,2,0,0,      NPYBase::ULONGLONG ,  OpticksBufferSpec::Get(sequence_, compute)) ;
    //    ULONGLONG -> RT_FORMAT_USER  and size set to ni*nj*nk = num_photons*1*2

    m_nopstep_spec = new NPYSpec(nopstep_   ,  0,4,4,0,0,      NPYBase::FLOAT     , OpticksBufferSpec::Get(nopstep_, compute) ) ;
    m_phosel_spec   = new NPYSpec(phosel_   ,  0,1,4,0,0,      NPYBase::UCHAR     , OpticksBufferSpec::Get(phosel_, compute) ) ;
    m_recsel_spec   = new NPYSpec(recsel_   ,  0,maxrec,1,4,0, NPYBase::UCHAR     , OpticksBufferSpec::Get(recsel_, compute) ) ;

    m_fdom_spec    = new NPYSpec(fdom_      ,  3,1,4,0,0,      NPYBase::FLOAT     ,  "" ) ;
    m_idom_spec    = new NPYSpec(idom_      ,  1,1,4,0,0,      NPYBase::INT       ,  "" ) ;

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
    const NPYSpec* spec = data->getBufferSpec();
    const char* name = data->getBufferName(); 

    if(!spec)
    {

        LOG(fatal) << "OpticksEvent::setBufferControl"
                     << " SKIPPED FOR " << name 
                     << " AS NO spec "
                     ;
#ifdef OLD_PARAMETERS
        X_BParameters* param = data->getParameters();
#else
        NMeta*       param = data->getParameters();
#endif
        if(param)
            param->dump("OpticksEvent::setBufferControl FATAL: BUFFER LACKS SPEC"); 
        assert(0);
        return ; 
    }


    // OpticksBufferControl argument is a pointer to 64-bit int 
    // living inside the NPYBase which has its contents 
    // defined by the below
    
    OpticksBufferControl ctrl(data->getBufferControlPtr());
    ctrl.add(spec->getCtrl());

    if(isCompute()) ctrl.add(OpticksBufferControl::COMPUTE_MODE_) ; 
    if(isInterop()) ctrl.add(OpticksBufferControl::INTEROP_MODE_) ; 

    if(ctrl("VERBOSE_MODE"))
     LOG(verbose) 
               << std::setw(10) << name 
               << " : " << ctrl.description("(spec)") 
               << " : " << brief()
               ;
}



void OpticksEvent::createBuffers(NPY<float>* gs)
{
    // invoked by Opticks::makeEvent 

    // NB allocation is deferred until zeroing and they start at 0 items anyhow
    //
    // NB by default gs = NULL and genstep buffer creation is excluded, 
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

    //NPY<float>* src = NPY<float>::make(m_source_spec);
    //setSourceData(src);   

    NPY<unsigned long long>* seq = NPY<unsigned long long>::make(m_sequence_spec); 
    setSequenceData(seq);   

    NPY<unsigned>* seed = NPY<unsigned>::make(m_seed_spec); 
    setSeedData(seed);   

    NPY<float>* hit = NPY<float>::make(m_hit_spec); 
    setHitData(hit);   

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
    // deallocate (clearing the underlying vector) and setNumItems to 0 
    if(m_nopstep_data)  m_nopstep_data->reset();    
    if(m_photon_data)   m_photon_data->reset();    
    if(m_sequence_data) m_sequence_data->reset();    
    if(m_seed_data)     m_seed_data->reset();    
    if(m_phosel_data)   m_phosel_data->reset();    
    if(m_recsel_data)   m_recsel_data->reset();    
    if(m_record_data)   m_record_data->reset();    
    if(m_hit_data)      m_hit_data->reset();    
}


/**
OpticksEvent::resize
---------------------

For dynamically recorded g4evt the photon, sequence and record 
buffers are grown during the instrumented Geant4 stepping, a
subsequent resize makes no difference to those buffers but pulls
up the counts for phosel and recsel (and seed) ready to 
hold the CPU indices. 

**/

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


    LOG(debug) << "OpticksEvent::resize " 
              << " num_photons " << num_photons  
              << " num_records " << num_records 
              << " maxrec " << maxrec
              << " " << getDir()
              ;

    m_photon_data->setNumItems(num_photons);
    m_sequence_data->setNumItems(num_photons);
    m_record_data->setNumItems(num_photons);

    m_seed_data->setNumItems(num_photons);
    m_phosel_data->setNumItems(num_photons);
    m_recsel_data->setNumItems(num_photons);

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


void OpticksEvent::setGenstepData(NPY<float>* genstep_data, bool progenitor)
{
    // gets called for OpticksRun::m_g4evt from OpticksRun::setGensteps by OKMgr::propagate 

    /**
    :param genstep_data:
    :param progenitor:
    :param oac_label:   adds to the OpticksActionControl to customize the import for different genstep types 
    **/


    int nitems = NPYBase::checkNumItems(genstep_data);
    if(nitems < 1)
    {
         LOG(warning) << "OpticksEvent::setGenstepData SKIP "
                      << " nitems " << nitems
                      ;
         return ; 
    } 


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
        bool resize_ = progenitor ; 
        setNumPhotons(num_photons, resize_); // triggers a resize   <<<<<<<<<<<<< SPECIAL HANDLING OF GENSTEP <<<<<<<<<<<<<<
    }
}

const glm::vec4& OpticksEvent::getGenstepCenterExtent()
{
    assert(m_genstep_vpos && "check hasGenstepData() before getGenstepCenterExtent"); 
    return m_genstep_vpos->getCenterExtent() ; 
}


bool OpticksEvent::isTorchType()
{    
   return strcmp(m_typ, OpticksFlags::torch_) == 0 ; 
}
bool OpticksEvent::isMachineryType()
{    
   return strcmp(m_typ, OpticksFlags::machinery_) == 0 ; 
}








OpticksBufferControl* OpticksEvent::getSeedCtrl()
{
   return m_seed_ctrl ; 
}
void OpticksEvent::setSeedData(NPY<unsigned>* seed_data)
{
    m_seed_data = seed_data  ;
    if(!seed_data)
    {
        LOG(debug) << "OpticksEvent::setSeedData seed_data NULL " ;
        return ; 
    }

    setBufferControl(seed_data);
    m_seed_ctrl = new OpticksBufferControl(m_seed_data->getBufferControlPtr());
    m_seed_attr = new MultiViewNPY("seed_attr");
}

void OpticksEvent::setHitData(NPY<float>* hit_data)
{
    m_hit_data = hit_data  ;
    if(!hit_data)
    {
        LOG(debug) << "OpticksEvent::setHitData hit_data NULL " ;
        return ; 
    }

    setBufferControl(hit_data);
    m_hit_attr = new MultiViewNPY("hit_attr");
}



OpticksBufferControl* OpticksEvent::getPhotonCtrl()
{
   return m_photon_ctrl ; 
}

OpticksBufferControl* OpticksEvent::getSourceCtrl()
{
   return m_source_ctrl ; 
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

void OpticksEvent::setSourceData(NPY<float>* source_data)
{
    if(!source_data) return ; 

    source_data->setBufferSpec(m_source_spec);  
    
    setBufferControl(source_data);

    m_source_data = source_data  ;
    m_source_ctrl = new OpticksBufferControl(m_source_data->getBufferControlPtr());
    if(m_num_source == 0) 
    {
        m_num_source = source_data->getShape(0) ;

        LOG(debug) << "OpticksEvent::setSourceData"
                  << " setting m_num_source from shape(0) " << m_num_source 
                  ;
    }
    else
    {
        assert(m_num_source == source_data->getShape(0));
    }

    //m_source_data->setDynamic();  // need to update with seeding so GL_DYNAMIC_DRAW needed 
    m_source_attr = new MultiViewNPY("source_attr");
    //                                                  j k l,sz   type          norm   iatt  item_from_dim
    m_source_attr->add(new ViewNPY("vpos",m_source_data,0,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 1st quad
    m_source_attr->add(new ViewNPY("vdir",m_source_data,1,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 2nd quad
    m_source_attr->add(new ViewNPY("vpol",m_source_data,2,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 3rd quad
    m_source_attr->add(new ViewNPY("iflg",m_source_data,3,0,0,4,ViewNPY::INT  , false, true , 1));      // 4th quad
}





void OpticksEvent::setNopstepData(NPY<float>* nopstep)
{
   /*
    int nitems = NPYBase::checkNumItems(nopstep);
    if(nitems < 1)
    {
         LOG(debug) << "OpticksEvent::setNopstepData SKIP "
                      << " nitems " << nitems
                      ;
         return ; 
    } 
   */



    m_nopstep_data = nopstep  ;
    if(!nopstep) return ; 

    setBufferControl(nopstep);

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
       << " num_source : " <<  m_num_source
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



void OpticksEvent::save()
{
    if(!CanAnalyse(this))
    {
        LOG(error) << "skip as CanAnalyse returns false (no rec) : " 
                   //<< getRelDir() 
                   ; 
        // This avoids writing the G4 evt domains, when running without --okg4 
        // that leads to unhealthy mixed timestamp event loads in evt.py. 
        // Different timestamps for ab.py between A and B 
        // is tolerated, although if too much time divergence is to be expected.
        return ; 
    }
    else
    {
        LOG(info) << "proceed as CanAnalyse returns true (with rec) : " 
                  //<< getRelDir()
                   ; 
    }


    OK_PROFILE("_OpticksEvent::save"); 


    LOG(info) << description("") << getShapeString() 
              << " dir " << m_event_spec->getDir()
               ;    

    if(m_ok->isProduction())
    {
        if(m_ok->hasOpt("savehit")) saveHitData();  // FOR production hit check
    }
    else
    {
        saveHitData();   
        saveNopstepData();
        saveGenstepData();
        savePhotonData();
        saveSourceData();
        saveRecordData();
        saveSequenceData();
        //saveSeedData();
        saveIndex();
    }

    recordDigests();
    saveDomains();
    saveParameters();

    OK_PROFILE("OpticksEvent::save"); 

    makeReport(false);  // after timer save, in order to include that in the report
    saveReport();
}



/**
OpticksEvent::saveHitData
--------------------------

Writes hit buffer even when empty, otherwise get inconsistent 
buffer time stamps when changes makes hits go away and are writing 
into the same directory.

Argument form allows externals like G4Opticks to save Geant4 sourced
hit data collected with CPhotonCollector into an event dir 
with minimal fuss. 

**/

void OpticksEvent::saveHitData() const 
{
    NPY<float>* ht = getHitData();
    saveHitData(ht); 
}

void OpticksEvent::saveHitData(NPY<float>* ht) const 
{
    if(ht)
    {
        unsigned num_hit = ht->getNumItems(); 
        ht->save(m_pfx, "ht", m_typ,  m_tag, m_udet);  // even when zero hits
        if(num_hit == 0) LOG(info) << "saveHitData zero hits " ; 
    }
}



void OpticksEvent::saveNopstepData()
{
    NPY<float>* no = getNopstepData();
    if(no)
    {
        unsigned num_nop = no->getNumItems(); 
        if(num_nop > 0)  no->save(m_pfx, "no", m_typ,  m_tag, m_udet);
        if(num_nop == 0) LOG(debug) << "saveNopstepData zero nop " ;
        if(num_nop > 0) no->dump("OpticksEvent::save (nopstep)");
    
        NPY<int>* idom = getIDomain();
        assert(idom && "OpticksEvent::save non-null nopstep BUT HAS NULL IDOM ");
    }
}
void OpticksEvent::saveGenstepData()
{
    // genstep were formally not saved as they exist already elsewhere,
    // however recording the gs in use for posterity makes sense
    // 
    NPY<float>* gs = getGenstepData();
    if(gs) gs->save(m_pfx, "gs", m_typ,  m_tag, m_udet);
}
void OpticksEvent::savePhotonData()
{
    NPY<float>* ox = getPhotonData();
    if(ox) ox->save(m_pfx, "ox", m_typ,  m_tag, m_udet);
}


void OpticksEvent::saveRecordData()
{
    NPY<short>* rx = getRecordData();    
    if(rx) rx->save(m_pfx, "rx", m_typ,  m_tag, m_udet);
}
void OpticksEvent::saveSequenceData()
{
    NPY<unsigned long long>* ph = getSequenceData();
    if(ph) ph->save(m_pfx, "ph", m_typ,  m_tag, m_udet);
}
void OpticksEvent::saveSeedData()
{
    // dont try, seed buffer is OPTIX_INPUT_ONLY , so attempts to download from GPU yields garbage
    // also suspect such an attempt messes up the OptiX context is peculiar ways 
    //
    // NPY<unsigned>* se  = getSeedData();
    // if(se) se->save(m_pfx, "se", m_typ,  m_tag, m_udet);
}



void OpticksEvent::saveSourceData() const 
{
    // source data originates CPU side, and is INPUT_ONLY to GPU side
    NPY<float>* so = getSourceData();
    saveSourceData(so); 
}

void OpticksEvent::saveSourceData(NPY<float>* so) const 
{
    if(so)
    {
        //unsigned num_so = so->getNumItems(); 
        so->save(m_pfx, "so", m_typ,  m_tag, m_udet);  
        //if(num_so == 0) LOG(info) << "saveSourceData zero source " ; 
    }
}



void OpticksEvent::makeReport(bool verbose)
{
    LOG(info) << "tagdir " << getTagDir()  ; 

    if(verbose)
    m_parameters->dump();

    m_report->add(m_versions->getLines());
    m_report->add(m_parameters->getLines());
    m_report->add(m_profile->getLines());

}


void OpticksEvent::saveReport()
{
    std::string tagdir = getTagDir();
    saveReport(tagdir.c_str());

    std::string anno = getTimeStamp() ;
    std::string tagdir_ts = getTagDir(anno.c_str());
    saveReport(tagdir_ts.c_str());
}



std::string OpticksEvent::TagDir(const char* pfx, const char* det, const char* typ, const char* tag, const char* anno)
{
    std::string tagdir = BOpticksEvent::directory(pfx, det, typ, tag, anno ? anno : NULL );
 
    return tagdir ; 

}
std::string OpticksEvent::getTagDir(const char* anno)
{
    const char* udet = getUDet();
    std::string tagdir = TagDir(m_pfx, udet, m_typ, m_tag, anno ? anno : NULL );


    if( anno == NULL )
    {
        const char* tagdir2 = getDir(); 
        assert( strcmp( tagdir.c_str(), tagdir2) == 0 );  
    }

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
    m_parameters->load(tagdir.c_str(), PARAMETERS_NAME );
}

void OpticksEvent::importParameters()
{
    std::string mode_ = m_parameters->get<std::string>("mode"); 
    OpticksMode* mode = new OpticksMode(mode_.c_str());
    LOG(debug) << "OpticksEvent::importParameters "
              << " mode_ " << mode_ 
              << " --> " << mode->description() ; 
    setMode(mode);
}



void OpticksEvent::saveReport(const char* dir)
{
    if(!m_report) return ; 
    LOG(debug) << "OpticksEvent::saveReport to " << dir  ; 

    m_profile->save(dir); 
    m_report->save(dir);  
    m_launch_times->save(dir);
    m_prelaunch_times->save(dir);

}

void OpticksEvent::loadReport()
{
    std::string tagdir = getTagDir();
    m_profile = OpticksProfile::Load( tagdir.c_str() );  
    m_report = Report::load(tagdir.c_str());
    m_launch_times = BTimes::Load(tagdir.c_str(), LAUNCH_LABEL );
    m_prelaunch_times = BTimes::Load(tagdir.c_str(), PRELAUNCH_LABEL );

}

void OpticksEvent::setFakeNopstepPath(const char* path)
{
    // fake path used by OpticksEvent::load rather than standard one
    // see npy-/nopstep_viz_debug.py

    m_fake_nopstep_path = path ? strdup(path) : NULL ;
}


OpticksEvent* OpticksEvent::load( const char* pfx, const char* typ, const char* tag, const char* det, const char* cat, bool verbose)
{
    LOG(info) 
        << " pfx " << pfx
        << " typ " << typ
        << " tag " << tag
        << " det " << det
        << " cat " << ( cat ? cat : "NULL" )
        ;

    OpticksEventSpec* spec = new OpticksEventSpec(pfx, typ, tag, det, cat);
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
    std::string path = BOpticksEvent::path(m_pfx, udet, m_typ, m_tag, name.c_str() );
    return strdup(path.c_str()) ; 
}


void OpticksEvent::importGenstepDataLoaded(NPY<float>* gs)
{
     OpticksActionControl ctrl(gs->getActionControlPtr());     
     ctrl.add(OpticksActionControl::GS_LOADED_);
     if(isTorchType())  ctrl.add(OpticksActionControl::GS_TORCH_);
}


/**
OpticksEvent::loadBuffers
---------------------------

TODO: move domain loading into separate method


**/

void OpticksEvent::loadBuffers(bool verbose)
{
    OK_PROFILE("_OpticksEvent::loadBuffers"); 

    const char* udet = getUDet(); // cat overrides det if present 

    bool qload = true ; 

    NPY<int>*   idom = NPY<int>::load(m_pfx, idom_, m_typ,  m_tag, udet, qload);

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

    NPY<float>* fdom = NPY<float>::load(m_pfx, fdom_, m_typ,  m_tag, udet, qload );

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
        no = NPY<float>::load(m_pfx, "no", m_typ,  m_tag, udet, qload);
    }
    if(no) loadBuffersImportSpec(no, m_nopstep_spec) ;

    NPY<float>*              gs = NPY<float>::load(m_pfx, "gs", m_typ,  m_tag, udet, qload);
    NPY<float>*              ox = NPY<float>::load(m_pfx, "ox", m_typ,  m_tag, udet, qload);
    NPY<short>*              rx = NPY<short>::load(m_pfx, "rx", m_typ,  m_tag, udet, qload);

    NPY<unsigned long long>* ph = NPY<unsigned long long>::load(m_pfx, "ph", m_typ,  m_tag, udet, qload );
    NPY<unsigned char>*      ps = NPY<unsigned char>::load(m_pfx, "ps", m_typ,  m_tag, udet, qload );
    NPY<unsigned char>*      rs = NPY<unsigned char>::load(m_pfx, "rs", m_typ,  m_tag, udet, qload );
    NPY<unsigned>*           se = NPY<unsigned>::load(     m_pfx, "se", m_typ,  m_tag, udet, qload );
    NPY<float>*              ht = NPY<float>::load(        m_pfx, "ht", m_typ,  m_tag, udet, qload );

    if(ph == NULL || ps == NULL || rs == NULL )
        LOG(warning) 
             << " " << getDir()
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
    if(ht) loadBuffersImportSpec(ht,m_hit_spec) ;

    if(gs) importGenstepDataLoaded(gs);   // sets action control, so setGenstepData label checks can succeed

    unsigned int num_genstep = gs ? gs->getShape(0) : 0 ;
    unsigned int num_nopstep = no ? no->getShape(0) : 0 ;
    unsigned int num_photons = ox ? ox->getShape(0) : 0 ;
    unsigned int num_history = ph ? ph->getShape(0) : 0 ;
    unsigned int num_phosel  = ps ? ps->getShape(0) : 0 ;
    unsigned int num_seed    = se ? se->getShape(0) : 0 ;
    unsigned int num_hit     = ht ? ht->getShape(0) : 0 ;

    // either zero or matching 
    assert(num_history == 0 || num_photons == num_history );
    assert(num_phosel == 0 || num_photons == num_phosel );
    assert(num_seed == 0 || num_photons == num_seed );
    

    unsigned int num_records = rx ? rx->getShape(0) : 0 ;
    unsigned int num_recsel  = rs ? rs->getShape(0) : 0 ;

    assert(num_recsel == 0 || num_records == num_recsel );


    LOG(debug) << "OpticksEvent::load shape(0) before reshaping "
              << " num_genstep " << num_genstep
              << " num_nopstep " << num_nopstep
              << " [ "
              << " num_photons " << num_photons
              << " num_history " << num_history
              << " num_phosel " << num_phosel 
              << " num_seed " << num_seed 
              << " num_hit " << num_hit
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
    setHitData(ht);

    OK_PROFILE("OpticksEvent::loadBuffers"); 

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
        if(ht) ht->Summary("ht");
    }

    if(!isIndexed())
    {
         LOG(warning) << "OpticksEvent::load IS NOT INDEXED " 
                      << brief()
                      ;
    }




}

bool OpticksEvent::isIndexed() const 
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
unsigned int OpticksEvent::getNumPhotonsPerG4Event() const 
{
   return m_parameters->get<int>("NumPhotonsPerG4Event","0");  // "0" : fallback if not set (eg for G4GUN running )
}




/**
OpticksEvent::postPropagateGeant4
----------------------------------

For dynamically recorded g4evt the photon, sequence and record 
buffers are grown during the instrumented Geant4 stepping, a
subsequent resize from setNumPhotons makes no difference to those buffers 
but pulls up the counts for phosel and recsel (and seed) ready to 
hold the CPU indices. 

**/
 
void OpticksEvent::postPropagateGeant4()
{
    unsigned int num_photons = m_photon_data->getShape(0);
    LOG(info) << "OpticksEvent::postPropagateGeant4"
              << " shape " << getShapeString()
              << " num_photons " << num_photons
              << " dynamic " << getDynamic() 
              ;

    //if(!m_ok->isFabricatedGensteps())    
    int dynamic = getDynamic(); 
    if(dynamic == 1)    
    {
        LOG(fatal) << " setting num_photons " << num_photons 
                   << " as dynamic : to pull up recsel, phosel ready to hold the indices " 
                   ; 
        setNumPhotons(num_photons);  
    }
    else
    {
        LOG(LEVEL) << " NOT setting num_photons, as STATIC running " << num_photons ; 
    }


    indexPhotonsCPU();    
    collectPhotonHitsCPU();
}

void OpticksEvent::collectPhotonHitsCPU()
{
    OK_PROFILE("_OpticksEvent::collectPhotonHitsCPU");

    NPY<float>* ox = getPhotonData();
    NPY<float>* ht = getHitData();

    unsigned hitmask = SURFACE_DETECT ;  
    // see notes/issues/geant4_opticks_integration/missing_cfg4_surface_detect.rst

    unsigned numHits = ox->write_selection(ht, 3,3, hitmask );

    LOG(info) << "OpticksEvent::collectPhotonHitsCPU"
              << " numHits " << numHits 
              ;

    OK_PROFILE("OpticksEvent::collectPhotonHitsCPU");
}


/**
OpticksEvent::indexPhotonsCPU
------------------------------

* used only for g4evt ie CRecorder/CWriter collected 
  records, photons, sequence buffers. 

* phosel and recsel are expected to have been sized, but 
  not to contain any data 

* unsigned long long sequence is the input to the indexing
  yielding phosel and recsel indices 


**/


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

    // hmm this is expecting a resize to have been done ???
    // in order to provide the slots in phosel and recsel 
    // for the index applyLookup to write to 

    if( sequence->getShape(0) != phosel->getShape(0))
    {
        LOG(fatal) << " length mismatch " 
                   << " sequence : " << sequence->getShape(0)
                   << " phosel   : " << phosel->getShape(0)
                   ;
               //    << " ABORT indexing "
         //return ;   
    }


    assert(sequence->getShape(0) == phosel->getShape(0));
    assert(sequence->getShape(0) == recsel0->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    LOG(info) << "indexSequence START " ;  
    idx->indexSequence(OpticksConst::SEQHIS_NAME_, OpticksConst::SEQMAT_NAME_);
    LOG(info) << "indexSequence DONE " ;  

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

    OK_PROFILE("OpticksEvent::indexPhotonsCPU");
}



void OpticksEvent::saveIndex()
{
    bool is_indexed = isIndexed();
    if(!is_indexed)
    {
        LOG(warning) << "SKIP as not indexed " ; 
        return ; 
    }

    NPY<unsigned char>* ps = getPhoselData();
    NPY<unsigned char>* rs = getRecselData();

    assert(ps);
    assert(rs);

    ps->save(m_pfx, "ps", m_typ,  m_tag, m_udet);
    rs->save(m_pfx, "rs", m_typ,  m_tag, m_udet);

    NPYBase::setGlobalVerbose(false);

    std::string tagdir = getTagDir();
    LOG(verbose) 
              << " tagdir " << tagdir
              << " seqhis " << m_seqhis
              << " seqmat " << m_seqmat
              << " bndidx " << m_bndidx
              ; 

    if(m_seqhis)
        m_seqhis->save(tagdir.c_str());        
    else
        LOG(LEVEL) << "no seqhis to save " ;

    if(m_seqmat)
        m_seqmat->save(tagdir.c_str());        
    else
        LOG(LEVEL) << "no seqmat to save " ;

    if(m_bndidx)
        m_bndidx->save(tagdir.c_str());        
    else
        LOG(LEVEL) << "no bndidx to save " ;
}

void OpticksEvent::loadIndex()
{
    std::string tagdir_ = getTagDir();
    const char* tagdir = tagdir_.c_str();
    const char* reldir = NULL ; 

    m_seqhis = Index::load(tagdir, OpticksConst::SEQHIS_NAME_, reldir );
    m_seqmat = Index::load(tagdir, OpticksConst::SEQMAT_NAME_, reldir );  
    m_bndidx = Index::load(tagdir, OpticksConst::BNDIDX_NAME_, reldir );

    LOG(debug) 
              << " tagdir " << tagdir 
              << " seqhis " << m_seqhis 
              << " seqmat " << m_seqmat 
              << " bndidx " << m_bndidx 
              ;
}



Index* OpticksEvent::loadNamedIndex( const char* pfx, const char* typ, const char* tag, const char* udet, const char* name)
{
    std::string tagdir = TagDir(pfx, udet, typ, tag);
    const char* reldir = NULL ; 
    Index* index = Index::load(tagdir.c_str(), name, reldir);

    if(!index)
    {
        LOG(warning) << "OpticksEvent::loadNamedIndex FAILED" 
                     << " name " << name
                     << " typ " << typ
                     << " tag " << tag
                     << " udet " << udet
                     << " tagdir " << tagdir 
                     ;
    }

    return index ; 
}

Index* OpticksEvent::loadHistoryIndex( const char* pfx, const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(pfx, typ, tag, udet, OpticksConst::SEQHIS_NAME_); 
}
Index* OpticksEvent::loadMaterialIndex( const char* pfx, const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(pfx, typ, tag, udet, OpticksConst::SEQMAT_NAME_); 
}
Index* OpticksEvent::loadBoundaryIndex( const char* pfx, const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(pfx, typ, tag, udet, OpticksConst::BNDIDX_NAME_); 
}


int OpticksEvent::seedDebugCheck(const char* msg)
{
    // This can only be used with specific debug entry points 
    // that write seeds as uint into the photon buffer
    //
    //     * entryCode T    TRIVIAL
    //     * entryCode D    DUMPSEED

    assert(m_photon_data && m_photon_data->hasData());
    assert(m_genstep_data && m_genstep_data->hasData());

    TrivialCheckNPY chk(m_photon_data, m_genstep_data, m_ok->getEntryCode());
    return chk.check(msg);
}




std::string OpticksEvent::getSeqHisString(unsigned photon_id) const 
{
    unsigned long long seqhis_ = getSeqHis(photon_id); 
    return OpticksFlags::FlagSequence(seqhis_);
}


unsigned long long OpticksEvent::getSeqHis(unsigned photon_id) const 
{
    unsigned num_seq = m_sequence_data ? m_sequence_data->getShape(0) : 0 ; 
    unsigned long long seqhis_ = photon_id < num_seq ?  m_sequence_data->getValue(photon_id,0,0) : 0 ;   
    return seqhis_ ; 
}

unsigned long long  OpticksEvent::getSeqMat(unsigned photon_id ) const 
{
    unsigned long long sm = m_sequence_data ? m_sequence_data->getValue(photon_id,0,1) : 0 ;
    return sm ;
}



