#pragma once
#include <map>
#include <string>
#include <cstring>
#include <cassert>

#include "NPY.hpp"

class Timer ; 
class Parameters ;
class Report ;
class TimesTable ; 

class Index ; 
class ViewNPY ;
class MultiViewNPY ;
class RecordsNPY ; 
class PhotonsNPY ; 


class OpticksEvent {
   public:
      static const char* PARAMETERS_NAME ;  
      static const char* TIMEFORMAT ;  
      static std::string timestamp();
   public:
      //    typ: cerenkov/scintillaton/torch/g4gun
      //    tag: 1/-1/2/-2/...  convention: -ve tags propagated by Geant4, +ve by Opticks
      //    det: dayabay/...    identifes the geocache  
      //    cat: OPTIONAL used for test categorization, eg PmtInBox
      static OpticksEvent* load(const char* typ, const char* tag, const char* det, const char* cat=NULL, bool verbose=false);
   public:
       OpticksEvent(const char* typ, const char* tag, const char* det, const char* cat=NULL);
       void setFlat(bool flat);
       void setStep(bool step);
   public:
       bool isFlat();
       bool isNoLoad();
       bool isLoaded();
       bool isIndexed();
       bool isStep();
   public:
       void indexPhotonsCPU();
   public:
       // from parameters
       unsigned int getBounceMax();
       unsigned int getRngMax();
       std::string getTimeStamp();
   private:
       void init();
   public:
       static const char* incoming ;
       static const char* primary ;
       static const char* genstep ;
       static const char* nopstep ;
       static const char* photon ;
       static const char* record  ;
       static const char* phosel ;
       static const char* recsel  ;
       static const char* sequence  ;
       static const char* aux ;
   public:
       NPY<float>* loadGenstepFromFile(int modulo=0);
       NPY<float>* loadGenstepDerivativeFromFile(const char* postfix="track", bool quietly=false);
       void setGenstepData(NPY<float>* genstep_data);
       void setNopstepData(NPY<float>* nopstep_data);
       void zero();
       void dumpDomains(const char* msg="OpticksEvent::dumpDomains");
       void prepareForIndexing();
       void prepareForPrimaryRecording();
       void resizeIndices();
   public:
       Parameters* getParameters();
       Timer*      getTimer();
       TimesTable* getTimesTable();
   public:
       void makeReport();
       void saveReport();
       void loadReport();
   private:
       void saveReport(const char* dir);
   public:
       void setIncomingData(NPY<float>* incoming_data);
   public:
       void setMaxRec(unsigned int maxrec);         // maximum record slots per photon
   public:
       // G4 related qtys used by cfg4- when OpticksEvent used to store G4 propagations
       void setNumG4Event(unsigned int n);
       void setNumPhotonsPerG4Event(unsigned int n);
       unsigned int getNumG4Event();
       unsigned int getNumPhotonsPerG4Event();
   public:
       void setBoundaryIndex(Index* bndidx);
       void setHistoryIndex(Index* seqhis);
       void setMaterialIndex(Index* seqmat);
       Index* getBoundaryIndex();
       Index* getHistoryIndex();
       Index* getMaterialIndex();
   public:
       // domains used for record compression
       void setSpaceDomain(const glm::vec4& space_domain);
       void setTimeDomain(const glm::vec4& time_domain);
       void setWavelengthDomain(const glm::vec4& wavelength_domain);
       const glm::vec4& getSpaceDomain();
       const glm::vec4& getTimeDomain();
       const glm::vec4& getWavelengthDomain();
   private:
       void updateDomainsBuffer();
       void importDomainsBuffer();
   public:
       void save(bool verbose=false);
       void saveIndex(bool verbose=false);
       void loadIndex();
       void loadBuffers(bool verbose=true);
   public: 
       void createBuffers(); 
   private:
       // invoked internally, as knock on from setGenstepData 
       void createHostBuffers(); 
       void createHostIndexBuffers(); 
       void createDomainBuffers();
       void createPhotonBuffers(unsigned int num_photons);
       void createFlatRecordBuffers(unsigned int num_records);
       void createStructuredRecordBuffers(unsigned int num_photons, unsigned int maxrec);
   private:
       void seedPhotonData();
       void setPrimaryData(NPY<float>* primary_data);
       void setPhotonData(NPY<float>* photon_data);
       void setSequenceData(NPY<unsigned long long>* history_data);
       void setRecordData(NPY<short>* record_data);
       void setAuxData(NPY<short>* aux_data);
       void setRecselData(NPY<unsigned char>* recsel_data);
       void setPhoselData(NPY<unsigned char>* phosel_data);
   public:
       static std::string speciesDir(const char* species, const char* udet, const char* typ);
   private:
       void recordDigests();
       std::string getSpeciesDir(const char* species); // tag in the name
       std::string getTagDir(const char* species, bool tstamp);     // tag in the dir 
       void saveParameters();
       void loadParameters();
   public:
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);
   public:
       NPY<float>*          getPrimaryData();
       NPY<float>*          getIncomingData();
       NPY<float>*          getGenstepData();
       NPY<float>*          getNopstepData();
       NPY<float>*          getPhotonData();
       NPY<short>*          getRecordData();
       NPY<short>*          getAuxData();
       NPY<unsigned char>*  getPhoselData();
       NPY<unsigned char>*  getRecselData();
       NPY<unsigned long long>*  getSequenceData();
   public:
       NPYBase*             getData(const char* name);
       std::string          getShapeString(); 
   public:
       // optionals lodged here for debug dumping single photons/records  
       void setRecordsNPY(RecordsNPY* recs);
       void setPhotonsNPY(PhotonsNPY* pho);
       RecordsNPY*          getRecordsNPY();
       PhotonsNPY*          getPhotonsNPY();
       NPY<float>*          getFDomain();
       NPY<int>*            getIDomain();
   public:
       void setFakeNopstepPath(const char* path);
   public:
       MultiViewNPY* getGenstepAttr();
       MultiViewNPY* getNopstepAttr();
       MultiViewNPY* getPhotonAttr();
       MultiViewNPY* getRecordAttr();
       MultiViewNPY* getAuxAttr();
       MultiViewNPY* getPhoselAttr();
       MultiViewNPY* getRecselAttr();
       MultiViewNPY* getSequenceAttr();

       ViewNPY* operator [](const char* spec);

   public:
       unsigned int getNumGensteps();
       unsigned int getNumNopsteps();
       unsigned int getNumPhotons();
       unsigned int getNumRecords();
       unsigned int getMaxRec();  // per-photon
   private:
       // set by setGenstepData based on summation over Cerenkov/Scintillation photons to generate
       void setNumPhotons(unsigned int num_photons);
   public:
       void Summary(const char* msg="OpticksEvent::Summary");
       std::string  description(const char* msg="OpticksEvent::description");
       void         dumpPhotonData();
       static void  dumpPhotonData(NPY<float>* photon_data);

       const char*  getTyp();
       const char*  getTag();
       const char*  getDet();
       const char*  getCat();
       const char*  getUDet();
   private:
       const char*           m_typ ; 
       const char*           m_tag ; 
       const char*           m_det ; 
       const char*           m_cat ; 
       bool                  m_flat ; 
       bool                  m_step ; 
       bool                  m_noload ; 
       bool                  m_loaded ; 

       Timer*                m_timer ;
       Parameters*           m_parameters ;
       Report*               m_report ;
       TimesTable*           m_ttable ;

       NPY<float>*           m_primary_data ; 
       NPY<float>*           m_incoming_data ; 
       NPY<float>*           m_genstep_data ;
       NPY<float>*           m_nopstep_data ;
       NPY<float>*           m_photon_data ;
       NPY<short>*           m_record_data ;
       NPY<short>*           m_aux_data ;
       NPY<unsigned char>*   m_phosel_data ;
       NPY<unsigned char>*   m_recsel_data ;
       NPY<unsigned long long>*  m_sequence_data ;

       NPY<float>*           m_fdom ; 
       NPY<int>*             m_idom ; 



       MultiViewNPY*   m_genstep_attr ;
       MultiViewNPY*   m_nopstep_attr ;
       MultiViewNPY*   m_photon_attr  ;
       MultiViewNPY*   m_record_attr  ;
       MultiViewNPY*   m_aux_attr  ;
       MultiViewNPY*   m_phosel_attr  ;
       MultiViewNPY*   m_recsel_attr  ;
       MultiViewNPY*   m_sequence_attr  ;

       RecordsNPY*     m_records ; 
       PhotonsNPY*     m_photons ; 

       unsigned int    m_num_gensteps ; 
       unsigned int    m_num_nopsteps ; 
       unsigned int    m_num_photons ; 
       unsigned int    m_maxrec ; 

       // hmm much of this can move to NPYBase ?

       glm::vec4       m_space_domain ; 
       glm::vec4       m_time_domain ; 
       glm::vec4       m_wavelength_domain ; 

       glm::ivec4      m_settings ; 

       Index*          m_seqhis ; 
       Index*          m_seqmat ; 
       Index*          m_bndidx ; 

       std::vector<std::string> m_data_names ; 
       std::map<std::string, std::string> m_abbrev ; 

       const char*     m_fake_nopstep_path ; 

};


inline OpticksEvent::OpticksEvent(const char* typ, const char* tag, const char* det, const char* cat) 
          :
          m_typ(strdup(typ)),
          m_tag(strdup(tag)),
          m_det(strdup(det)),
          m_cat(strdup(cat)),
          m_flat(false),
          m_step(true),
          m_noload(false),
          m_loaded(false),

          m_timer(NULL),
          m_parameters(NULL),
          m_report(NULL),
          m_ttable(NULL),

          m_primary_data(NULL),
          m_incoming_data(NULL),
          m_genstep_data(NULL),
          m_nopstep_data(NULL),
          m_photon_data(NULL),
          m_record_data(NULL),
          m_aux_data(NULL),
          m_phosel_data(NULL),
          m_recsel_data(NULL),
          m_sequence_data(NULL),

          m_fdom(NULL),
          m_idom(NULL),

          m_genstep_attr(NULL),
          m_nopstep_attr(NULL),
          m_photon_attr(NULL),
          m_record_attr(NULL),
          m_aux_attr(NULL),
          m_phosel_attr(NULL),
          m_recsel_attr(NULL),
          m_sequence_attr(NULL),

          m_records(NULL),
          m_photons(NULL),
          m_num_gensteps(0),
          m_num_nopsteps(0),
          m_num_photons(0),
          m_maxrec(1),
          m_seqhis(NULL),
          m_seqmat(NULL),
          m_bndidx(NULL),
          m_fake_nopstep_path(NULL)
{
    init();
}


inline void OpticksEvent::setFlat(bool flat)
{
    m_flat = flat ;
}
inline bool OpticksEvent::isFlat()
{
    return m_flat ; 
}

inline void OpticksEvent::setStep(bool step)
{
    m_step = step ;
}
inline bool OpticksEvent::isStep()
{
    return m_step ; 
}



inline bool OpticksEvent::isNoLoad()
{
    return m_noload ; 
}

inline bool OpticksEvent::isLoaded()
{
    return m_loaded ; 
}




inline unsigned int OpticksEvent::getNumGensteps()
{
    return m_num_gensteps ; 
}
inline unsigned int OpticksEvent::getNumNopsteps()
{
    return m_num_nopsteps ; 
}

inline void OpticksEvent::setNumPhotons(unsigned int num_photons)
{
    m_num_photons = num_photons ; 
}
inline unsigned int OpticksEvent::getNumPhotons()
{
    return m_num_photons ; 
}


inline unsigned int OpticksEvent::getNumRecords()
{
    return m_num_photons * m_maxrec ; 
}
inline unsigned int OpticksEvent::getMaxRec()
{
    return m_maxrec ; 
}
inline void OpticksEvent::setMaxRec(unsigned int maxrec)
{
    m_maxrec = maxrec ; 
    m_settings.w = m_maxrec ; 
}





inline NPY<float>* OpticksEvent::getGenstepData()
{
    return m_genstep_data ;
}
inline MultiViewNPY* OpticksEvent::getGenstepAttr()
{
    return m_genstep_attr ;
}

inline NPY<float>* OpticksEvent::getNopstepData()
{
    return m_nopstep_data ;
}
inline MultiViewNPY* OpticksEvent::getNopstepAttr()
{
    return m_nopstep_attr ;
}





inline NPY<float>* OpticksEvent::getPhotonData()
{
    return m_photon_data ;
}

inline MultiViewNPY* OpticksEvent::getPhotonAttr()
{
    return m_photon_attr ;
}

inline NPY<short>* OpticksEvent::getRecordData()
{
    return m_record_data ;
}
inline MultiViewNPY* OpticksEvent::getRecordAttr()
{
    return m_record_attr ;
}

inline NPY<short>* OpticksEvent::getAuxData()
{
    return m_aux_data ;
}
inline MultiViewNPY* OpticksEvent::getAuxAttr()
{
    return m_aux_attr ;
}





inline NPY<unsigned char>* OpticksEvent::getPhoselData()
{
    return m_phosel_data ;
}
inline MultiViewNPY* OpticksEvent::getPhoselAttr()
{
    return m_phosel_attr ;
}


inline NPY<unsigned char>* OpticksEvent::getRecselData()
{
    return m_recsel_data ;
}
inline MultiViewNPY* OpticksEvent::getRecselAttr()
{
    return m_recsel_attr ;
}




inline NPY<unsigned long long>* OpticksEvent::getSequenceData()
{
    return m_sequence_data ;
}
inline MultiViewNPY* OpticksEvent::getSequenceAttr()
{
    return m_sequence_attr ;
}





inline NPY<float>* OpticksEvent::getPrimaryData()
{
    return m_primary_data ;
}
inline void OpticksEvent::setPrimaryData(NPY<float>* primary_data)
{
    m_primary_data = primary_data ;
}





inline NPY<float>* OpticksEvent::getIncomingData()
{
    return m_incoming_data ;
}

inline void OpticksEvent::setIncomingData(NPY<float>* incoming_data)
{
    m_incoming_data = incoming_data ;
}

inline void OpticksEvent::setRecordsNPY(RecordsNPY* records)
{
    m_records = records ; 
}
inline RecordsNPY* OpticksEvent::getRecordsNPY()
{
    return m_records ;
}

inline void OpticksEvent::setPhotonsNPY(PhotonsNPY* photons)
{
    m_photons = photons ; 
}
inline PhotonsNPY* OpticksEvent::getPhotonsNPY()
{
    return m_photons ;
}




inline void OpticksEvent::setFDomain(NPY<float>* fdom)
{
    m_fdom = fdom ; 
}
inline void OpticksEvent::setIDomain(NPY<int>* idom)
{
    m_idom = idom ; 
}

inline NPY<float>* OpticksEvent::getFDomain()
{
    return m_fdom ; 
}
inline NPY<int>* OpticksEvent::getIDomain()
{
    return m_idom ; 
}

inline const char* OpticksEvent::getTyp()
{
    return m_typ ; 
}
inline const char* OpticksEvent::getTag()
{
    return m_tag ; 
}
inline const char* OpticksEvent::getDet()
{
    return m_det ; 
}
inline const char* OpticksEvent::getCat()
{
    return m_cat ; 
}
inline const char* OpticksEvent::getUDet()
{
    return strlen(m_cat) > 0 ? m_cat : m_det ; 
}




inline void OpticksEvent::setSpaceDomain(const glm::vec4& space_domain)
{
    m_space_domain = space_domain ; 
}
inline void OpticksEvent::setTimeDomain(const glm::vec4& time_domain)
{
    m_time_domain = time_domain  ; 
}
inline void OpticksEvent::setWavelengthDomain(const glm::vec4& wavelength_domain)
{
    m_wavelength_domain = wavelength_domain  ; 
}


inline const glm::vec4& OpticksEvent::getSpaceDomain()
{
    return m_space_domain ; 
}
inline const glm::vec4& OpticksEvent::getTimeDomain()
{
    return m_time_domain ;
}
inline const glm::vec4& OpticksEvent::getWavelengthDomain()
{ 
    return m_wavelength_domain ; 
}

inline void OpticksEvent::setBoundaryIndex(Index* bndidx)
{
    // called from OpIndexer::indexBoundaries
    m_bndidx = bndidx ; 
}
inline void OpticksEvent::setHistoryIndex(Index* seqhis)
{
    // called from OpIndexer::indexSequenceLoaded 
    m_seqhis = seqhis ; 
}
inline void OpticksEvent::setMaterialIndex(Index* seqmat)
{
    // called from OpIndexer::indexSequenceLoaded
    m_seqmat = seqmat ; 
}


inline Index* OpticksEvent::getHistoryIndex()
{
    return m_seqhis ; 
} 
inline Index* OpticksEvent::getMaterialIndex()
{
    return m_seqmat ; 
} 
inline Index* OpticksEvent::getBoundaryIndex()
{
    return m_bndidx ; 
}




inline Parameters* OpticksEvent::getParameters()
{
    return m_parameters ;
}
inline Timer* OpticksEvent::getTimer()
{
    return m_timer ;
}
inline TimesTable* OpticksEvent::getTimesTable()
{
    return m_ttable ;
}




  
