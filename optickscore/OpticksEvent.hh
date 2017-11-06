

#include <map>
#include <string>

//template <typename T> class NPY ; 
#include "NPY.hpp"

struct STimes ; 

class Timer ; 
class NParameters ;
class Report ;
class TimesTable ; 


class Index ; 
class ViewNPY ;
class MultiViewNPY ;
class RecordsNPY ; 
class PhotonsNPY ; 
class BoundariesNPY ; 
//class G4StepNPY ; 
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


TODO:

* currently switching to production mode
  which persists much less results in a mixed write event, 
  current approach 
  of persisting evt metadata into the evt tag folder makes
  it not so easy to just scrub the event

  * TODO: make metadata folders sibling to the evt ?
          so can easily scrub ?  


**/

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

#include "OpticksEventSpec.hh"

class OKCORE_API OpticksEvent : public OpticksEventSpec 
{
      // loadBuffers
      friend class Opticks ; 
      friend class OpticksRun ; 
      // saveIndex
      friend class OpIndexerApp ; 
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
      static NPYSpec* GenstepSpec(bool compute);
      static NPYSpec* SourceSpec(bool compute);
      static NPYSpec* SeedSpec(bool compute);
      static void pushNames(std::vector<std::string>& names);
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
       Opticks*       getOpticks();
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
       char getEntryCode();

       void setTimeStamp(const char* tstamp);
       void setCreator(const char* executable);
       void setEntryCode(char entryCode);
   public:
        const char* getGeoPath();
   private:
       std::string getTestCSGPath();
       void setTestCSGPath(const char* testcsgpath);
   private:
       void setRngMax(unsigned int rng_max);
       void init();
       void indexPhotonsCPU();
       void collectPhotonHitsCPU();
   public:
       static const char* idom_ ;
       static const char* fdom_ ;
       static const char* genstep_ ;
       static const char* nopstep_ ;
       static const char* photon_ ;
       static const char* source_ ;
       static const char* record_  ;
       static const char* phosel_ ;
       static const char* recsel_  ;
       static const char* sequence_  ;
       static const char* seed_  ;
       static const char* hit_  ;
   public:
       NPY<float>* loadGenstepDerivativeFromFile(const char* stem="track");
       void setGenstepData(NPY<float>* genstep_data, bool progenitor=true);
       void setNopstepData(NPY<float>* nopstep_data);


       void zero();
       void dumpDomains(const char* msg="OpticksEvent::dumpDomains");
   public:
       void addBufferControl(const char* name, const char* ctrl_);
       int seedDebugCheck(const char* msg="OpticksEvent::seedDebugCheck");
  private:
       void setBufferControl(NPYBase* data);

   public:
       const char* getPath(const char* xx);  // accepts abbreviated or full constituent names
       NParameters* getParameters();
       Timer*      getTimer();
       TimesTable* getTimesTable();
   public:
       void makeReport(bool verbose=false);
       void saveReport();
       void loadReport();
   private:
       void saveDomains();
       void saveReport(const char* dir);
   public:
       void setMaxRec(unsigned maxrec);         // maximum record slots per photon
       void setMaxBounce(unsigned maxbounce);         // maximum record slots per photon
       void setMaxRng(unsigned maxrng); 
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
       void save();
   private:
       void saveHitData(); 
       void saveNopstepData(); 
       void saveGenstepData(); 
       void savePhotonData(); 
       void saveSourceData(); 
       void saveRecordData(); 
       void saveSequenceData(); 
       void saveSeedData(); 
       void saveIndex();
       void loadIndex();
       void loadBuffers(bool verbose=true);

   private:
       void importGenstepDataLoaded(NPY<float>* gs);
       void loadBuffersImportSpec(NPYBase* npy, NPYSpec* spec);
   public: 
       void createBuffers(NPY<float>* gs=NULL); 
       void createSpec(); 
   private:
       void setPhotonData(NPY<float>* photon_data);
       void setSourceData(NPY<float>* source_data);
       void setSequenceData(NPY<unsigned long long>* history_data);
       void setSeedData(NPY<unsigned>* seed_data);
       void setHitData(NPY<float>* hit_data);
       void setRecordData(NPY<short>* record_data);
       void setRecselData(NPY<unsigned char>* recsel_data);
       void setPhoselData(NPY<unsigned char>* phosel_data);
   private:
       void recordDigests();
   public:
       std::string getTagDir(const char* anno=NULL); // anno usually NULL, sometimes the timestamp
       static std::string TagDir(const char* det, const char* typ, const char* tag, const char* anno=NULL);
   public:
       unsigned long long getSeqHis(unsigned photon_id) const ; 
       unsigned long long getSeqMat(unsigned photon_id) const ; 
   private:
       void saveParameters();
       void loadParameters();
       void importParameters();
   public:
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);
   public:
       bool                 hasPhotonData();
       bool                 hasSourceData();
       bool                 hasGenstepData();
       const glm::vec4&     getGenstepCenterExtent();
   public:
       NPY<float>*          getGenstepData();
       NPY<float>*          getNopstepData();
       NPY<float>*          getPhotonData();
       NPY<float>*          getSourceData();
       NPY<short>*          getRecordData();
       NPY<unsigned char>*  getPhoselData();
       NPY<unsigned char>*  getRecselData();
       NPY<unsigned long long>*  getSequenceData();
       NPY<unsigned>*          getSeedData();
       NPY<float>*             getHitData();
   public:
       OpticksBufferControl* getPhotonCtrl();
       OpticksBufferControl* getSourceCtrl();
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
       MultiViewNPY* getSourceAttr();
       MultiViewNPY* getRecordAttr();
       MultiViewNPY* getPhoselAttr();
       MultiViewNPY* getRecselAttr();
       MultiViewNPY* getSequenceAttr();
       MultiViewNPY* getSeedAttr();
       MultiViewNPY* getHitAttr();

       ViewNPY* operator [](const char* spec);

   public:
       unsigned int getNumGensteps();
       unsigned int getNumNopsteps();
       unsigned int getNumPhotons();
       unsigned int getNumSource();
       unsigned int getNumRecords();
   public:
       // idom persisted 
       unsigned int getMaxRec();  // per-photon
       unsigned int getMaxBounce();  // per-photon
       unsigned int getMaxRng();  
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

       bool         isInterop();
       bool         isCompute();

   private:
       OpticksEventSpec*     m_event_spec ; 
       Opticks*              m_ok ;  
       OpticksMode*          m_mode ; 

       bool                  m_noload ; 
       bool                  m_loaded ; 

       Timer*                m_timer ;
       NParameters*           m_parameters ;
       Report*               m_report ;
       TimesTable*           m_ttable ;

       NPY<float>*           m_primary_data ; 
       NPY<float>*           m_genstep_data ;
       NPY<float>*           m_nopstep_data ;
       NPY<float>*           m_photon_data ;
       NPY<float>*           m_source_data ;
       NPY<short>*           m_record_data ;
       NPY<unsigned char>*   m_phosel_data ;
       NPY<unsigned char>*   m_recsel_data ;
       NPY<unsigned long long>*  m_sequence_data ;
       NPY<unsigned>*           m_seed_data ;
       NPY<float>*              m_hit_data ;

       OpticksBufferControl*  m_photon_ctrl ; 
       OpticksBufferControl*  m_source_ctrl ; 
       OpticksBufferControl*  m_seed_ctrl ; 
       OpticksDomain*        m_domain ; 

       //G4StepNPY*      m_g4step ; 
       ViewNPY*        m_genstep_vpos ;

       MultiViewNPY*   m_genstep_attr ;
       MultiViewNPY*   m_nopstep_attr ;
       MultiViewNPY*   m_photon_attr  ;
       MultiViewNPY*   m_source_attr  ;
       MultiViewNPY*   m_record_attr  ;
       MultiViewNPY*   m_phosel_attr  ;
       MultiViewNPY*   m_recsel_attr  ;
       MultiViewNPY*   m_sequence_attr  ;
       MultiViewNPY*   m_seed_attr  ;
       MultiViewNPY*   m_hit_attr  ;

       RecordsNPY*     m_records ; 
       PhotonsNPY*     m_photons ; 
       HitsNPY*        m_hits ; 
       BoundariesNPY*  m_bnd ; 

       unsigned int    m_num_gensteps ; 
       unsigned int    m_num_nopsteps ; 
       unsigned int    m_num_photons ; 
       unsigned int    m_num_source ; 

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
       NPYSpec* m_source_spec ;  
       NPYSpec* m_record_spec ;  
       NPYSpec* m_phosel_spec ;  
       NPYSpec* m_recsel_spec ;  
       NPYSpec* m_sequence_spec ;  
       NPYSpec* m_seed_spec ;  
       NPYSpec* m_hit_spec ;  

       STimes*  m_prelaunch_times ; 
       STimes*  m_launch_times ; 

       OpticksEvent*  m_sibling ; 
       const char*    m_geopath ; 

};

//
// avoiding class members simplifies usage, as full type spec is not needed for pointers : forward declarations sufficient
// for this reason moved glm domain vector members down into OpticksDomain
// this simplifies use with nvcc compiler
// 

#include "OKCORE_TAIL.hh"
  
