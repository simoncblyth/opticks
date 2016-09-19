

#include <map>
#include <string>

//template <typename T> class NPY ; 
#include "NPY.hpp"

struct STimes ; 

class Timer ; 
class Parameters ;
class Report ;
class TimesTable ; 


class Index ; 
class ViewNPY ;
class MultiViewNPY ;
class RecordsNPY ; 
class PhotonsNPY ; 
class BoundariesNPY ; 
class G4StepNPY ; 
class HitsNPY ; 
class NPYSpec ; 

class Opticks ; 
class OpticksMode ; 
class OpticksBufferControl ; 
class OpticksDomain ; 

/**
OpticksEvent
=============

NPY buffer allocation on the host is deferred until/if they are downloaded from the GPU. 
The buffer shapes represent future sizes if they ever get downloaded. 
Only the generally small gensteps and nopsteps are usually hostside allocated, 
as they are the input buffers.

So there is no problem with having multiple OpticksEvent instances.
But full GPU memory is immediately allocated on "uploading", 
so avoid uploading more than one.


nopstep
      non-optical steps
genstep
      scintillation or cerenkov
records
      photon step records
photons
      last photon step at absorption, detection
sequence   
      photon level material/flag histories
phosel
      obtained by indexing *sequence*
recsel
      obtained by repeating *phosel* by maxrec

**/

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

#include "OpticksEventSpec.hh"

class OKCORE_API OpticksEvent : public OpticksEventSpec {
   public:
      static const char* PARAMETERS_NAME ;  
      static const char* PARAMETERS_STEM ;  
      static const char* PARAMETERS_EXT ;  
      static const char* TIMEFORMAT ;  
      static std::string timestamp();
   public:
      //
      //    typ: cerenkov/scintillaton/torch/g4gun
      //    tag: 1/-1/2/-2/...  convention: -ve tags propagated by Geant4, +ve by Opticks
      //    det: dayabay/...    identifes the geocache  
      //    cat: optional override of det for test categorization, eg PmtInBox
      //
      static OpticksEvent* load(const char* typ, const char* tag, const char* det, const char* cat=NULL, bool verbose=false);
      static Index* loadHistoryIndex(  const char* typ, const char* tag, const char* udet);
      static Index* loadMaterialIndex( const char* typ, const char* tag, const char* udet);
      static Index* loadBoundaryIndex( const char* typ, const char* tag, const char* udet);
      static Index* loadNamedIndex(    const char* typ, const char* tag, const char* udet, const char* name);
      static NPYSpec* GenstepSpec();
      static NPYSpec* SeedSpec();
   public:
       static OpticksEvent* make(OpticksEventSpec* spec, unsigned tagoffset=0);
       OpticksEvent(OpticksEventSpec* spec);
       void reset();
       virtual ~OpticksEvent();
   private:
       void resetBuffers();
   public:
       // set by Opticks::makeEvent OpticksRun::createEvent
       void           setSibling(OpticksEvent* sibling);
       void           setOpticks(Opticks* ok);
       void           setId(int id);
   public:
       OpticksEvent*  getSibling();
       int  getId();
   public:
       bool isNoLoad();
       bool isLoaded();
       bool isIndexed();
       bool isStep();
       bool isFlat();
       bool isTorchType();
       bool isMachineryType();

       STimes* getPrelaunchTimes();
       STimes* getLaunchTimes();
   public:
       void postPropagateGeant4(); // called following dynamic photon/record/sequence collection
   public:
       // from parameters set in Opticks::makeEvent
       unsigned int getBounceMax();
       unsigned int getRngMax();
       std::string getTimeStamp();
       std::string getCreator();

       void setTimeStamp(const char* tstamp);
       void setCreator(const char* executable);
   private:
       void setRngMax(unsigned int rng_max);
       void init();
       void indexPhotonsCPU();
   public:
       static const char* idom_ ;
       static const char* fdom_ ;
       static const char* genstep_ ;
       static const char* nopstep_ ;
       static const char* photon_ ;
       static const char* record_  ;
       static const char* phosel_ ;
       static const char* recsel_  ;
       static const char* sequence_  ;
       static const char* seed_  ;
   public:
       NPY<float>* loadGenstepDerivativeFromFile(const char* stem="track");
       void setGenstepData(NPY<float>* genstep_data, bool progenitor=true, const char* oac_label=NULL);
       void setNopstepData(NPY<float>* nopstep_data);


       G4StepNPY* getG4Step(); 
       void zero();
       void dumpDomains(const char* msg="OpticksEvent::dumpDomains");
   public:
       void addBufferControl(const char* name, const char* ctrl_);
   private:
       void importGenstepDataLoaded(NPY<float>* gs);
       void importGenstepData(NPY<float>* gs, const char* oac_label=NULL);
       void translateLegacyGensteps(NPY<float>* gs);
       void setBufferControl(NPYBase* data);

   public:
       const char* getPath(const char* xx);  // accepts abbreviated or full constituent names
       Parameters* getParameters();
       Timer*      getTimer();
       TimesTable* getTimesTable();
   public:
       void makeReport(bool verbose=false);
       void saveReport();
       void loadReport();
   private:
       void saveReport(const char* dir);
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
       // below are set in Opticks::makeEvent   
       void setMode(OpticksMode* mode); 
       // domains used for record compression
       void setSpaceDomain(const glm::vec4& space_domain);
       void setTimeDomain(const glm::vec4& time_domain);
       void setWavelengthDomain(const glm::vec4& wavelength_domain);
   public:
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
   private:
       void loadBuffersImportSpec(NPYBase* npy, NPYSpec* spec);
   public: 
       void createBuffers(NPY<float>* gs=NULL); 
       void createSpec(); 
   private:
       void setPhotonData(NPY<float>* photon_data);
       void setSequenceData(NPY<unsigned long long>* history_data);
       void setSeedData(NPY<unsigned>* seed_data);
       void setRecordData(NPY<short>* record_data);
       void setRecselData(NPY<unsigned char>* recsel_data);
       void setPhoselData(NPY<unsigned char>* phosel_data);
   private:
       void recordDigests();
   public:
       std::string getTagDir(const char* anno=NULL); // anno usually NULL, sometimes the timestamp
   private:
       static std::string TagDir(const char* det, const char* typ, const char* tag, const char* anno=NULL);
   private:
       void saveParameters();
       void loadParameters();
       void importParameters();
   public:
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);
   public:
       bool                 hasPhotonData();
       bool                 hasGenstepData();
       const glm::vec4&     getGenstepCenterExtent();
   public:
       NPY<float>*          getGenstepData();
       NPY<float>*          getNopstepData();
       NPY<float>*          getPhotonData();
       NPY<short>*          getRecordData();
       NPY<unsigned char>*  getPhoselData();
       NPY<unsigned char>*  getRecselData();
       NPY<unsigned long long>*  getSequenceData();
       NPY<unsigned>*          getSeedData();
   public:
       OpticksBufferControl* getPhotonCtrl();
       OpticksBufferControl* getSeedCtrl();
   public:
       NPYBase*             getData(const char* name);
       NPYSpec*             getSpec(const char* name);
       std::string          getShapeString(); 
   public:
       // optionals lodged here for debug dumping single photons/records  
       void setRecordsNPY(RecordsNPY* recs);
       void setPhotonsNPY(PhotonsNPY* pho);
       void setHitsNPY(HitsNPY* hit);
       void setBoundariesNPY(BoundariesNPY* bnd);

       RecordsNPY*          getRecordsNPY();
       PhotonsNPY*          getPhotonsNPY();
       HitsNPY*             getHitsNPY();
       BoundariesNPY*       getBoundariesNPY();
   public:
       NPY<float>*          getFDomain();
       NPY<int>*            getIDomain();
   public:
       void setFakeNopstepPath(const char* path);
   public:
       MultiViewNPY* getGenstepAttr();
       MultiViewNPY* getNopstepAttr();
       MultiViewNPY* getPhotonAttr();
       MultiViewNPY* getRecordAttr();
       MultiViewNPY* getPhoselAttr();
       MultiViewNPY* getRecselAttr();
       MultiViewNPY* getSequenceAttr();
       MultiViewNPY* getSeedAttr();

       ViewNPY* operator [](const char* spec);

   public:
       unsigned int getNumGensteps();
       unsigned int getNumNopsteps();
       unsigned int getNumPhotons();
       unsigned int getNumRecords();
       unsigned int getMaxRec();  // per-photon
   public:
       void resizeToZero();  // used by OpticksHub::setupZeroEvent
   private:
       // set by setGenstepData based on summation over Cerenkov/Scintillation photons to generate
       void setNumPhotons(unsigned int num_photons, bool resize=true);
       void resize();
   public:
       void Summary(const char* msg="OpticksEvent::Summary");
       std::string  brief();
       std::string  description(const char* msg="OpticksEvent::description");
       void         dumpPhotonData();
       static void  dumpPhotonData(NPY<float>* photon_data);

       bool         isInterop();
       bool         isCompute();

   private:
       OpticksEventSpec*     m_event_spec ; 
       Opticks*              m_ok ;  
       OpticksMode*          m_mode ; 

       bool                  m_noload ; 
       bool                  m_loaded ; 

       Timer*                m_timer ;
       Parameters*           m_parameters ;
       Report*               m_report ;
       TimesTable*           m_ttable ;

       NPY<float>*           m_primary_data ; 
       NPY<float>*           m_genstep_data ;
       NPY<float>*           m_nopstep_data ;
       NPY<float>*           m_photon_data ;
       NPY<short>*           m_record_data ;
       NPY<unsigned char>*   m_phosel_data ;
       NPY<unsigned char>*   m_recsel_data ;
       NPY<unsigned long long>*  m_sequence_data ;
       NPY<unsigned>*           m_seed_data ;

       OpticksBufferControl*  m_photon_ctrl ; 
       OpticksBufferControl*  m_seed_ctrl ; 
       OpticksDomain*        m_domain ; 

       G4StepNPY*      m_g4step ; 
       ViewNPY*        m_genstep_vpos ;

       MultiViewNPY*   m_genstep_attr ;
       MultiViewNPY*   m_nopstep_attr ;
       MultiViewNPY*   m_photon_attr  ;
       MultiViewNPY*   m_record_attr  ;
       MultiViewNPY*   m_phosel_attr  ;
       MultiViewNPY*   m_recsel_attr  ;
       MultiViewNPY*   m_sequence_attr  ;
       MultiViewNPY*   m_seed_attr  ;

       RecordsNPY*     m_records ; 
       PhotonsNPY*     m_photons ; 
       HitsNPY*        m_hits ; 
       BoundariesNPY*  m_bnd ; 

       unsigned int    m_num_gensteps ; 
       unsigned int    m_num_nopsteps ; 
       unsigned int    m_num_photons ; 

       Index*          m_seqhis ; 
       Index*          m_seqmat ; 
       Index*          m_bndidx ; 

       std::vector<std::string>           m_data_names ; 
       std::map<std::string, std::string> m_abbrev ; 

       const char*     m_fake_nopstep_path ; 

       NPYSpec* m_fdom_spec ;  
       NPYSpec* m_idom_spec ;  
       NPYSpec* m_genstep_spec ;  
       NPYSpec* m_nopstep_spec ;  
       NPYSpec* m_photon_spec ;  
       NPYSpec* m_record_spec ;  
       NPYSpec* m_phosel_spec ;  
       NPYSpec* m_recsel_spec ;  
       NPYSpec* m_sequence_spec ;  
       NPYSpec* m_seed_spec ;  

       STimes*  m_prelaunch_times ; 
       STimes*  m_launch_times ; 

       OpticksEvent*  m_sibling ; 

};

//
// avoiding class members simplifies usage, as full type spec is not needed for pointers : forward declarations sufficient
// for this reason moved glm domain vector members down into OpticksDomain
// this simplifies use with nvcc compiler
// 

#include "OKCORE_TAIL.hh"
  
