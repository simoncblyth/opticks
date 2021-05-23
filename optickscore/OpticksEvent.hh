/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */



#include <map>
#include <vector>
#include <string>

#include "plog/Severity.h"
#include "NGLM.hpp"

class BMeta ; 
class NPYBase ; 
template <typename T> class NPY ; 

class BTimes ; 
class BTimesTable ; 
class BTimeKeeper ; 

class Report ;


class Index ; 
class ViewNPY ;
class MultiViewNPY ;
class RecordsNPY ; 
class PhotonsNPY ; 
class BoundariesNPY ; 
class NPYSpec ; 
class NGeoTestConfig ; 

class Opticks ; 
class OpticksProfile ; 
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
      static const plog::Severity LEVEL ; 

      // loadBuffers
      friend class Opticks ; 
      friend class OpticksRun ; 
      // saveIndex
      friend class OpIndexerApp ; 
      // setNumPhotons
      friend struct CManager ; 
   public:
      static const char* PRELAUNCH_LABEL ;  
      static const char* LAUNCH_LABEL ;  
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
      static bool CanAnalyse(OpticksEvent* evt); 
      static OpticksEvent* load(       const char* pfx, const char* typ, const char* tag, const char* det, const char* cat=NULL, bool verbose=false);
      static Index* loadHistoryIndex(  const char* pfx, const char* typ, const char* tag, const char* udet);
      static Index* loadMaterialIndex( const char* pfx, const char* typ, const char* tag, const char* udet);
      static Index* loadBoundaryIndex( const char* pfx, const char* typ, const char* tag, const char* udet);
      static Index* loadNamedIndex(    const char* pfx, const char* typ, const char* tag, const char* udet, const char* name);
      static NPYSpec* GenstepSpec(bool compute);
      static NPYSpec* SourceSpec(bool compute);
      static NPYSpec* SeedSpec(bool compute);
      static void pushNames(std::vector<std::string>& names);
   public:
       static OpticksEvent* Make(OpticksEventSpec* spec, unsigned tagoffset=0);
       OpticksEvent(OpticksEventSpec* spec);
       void reset();
       virtual ~OpticksEvent();
   private:
       void resetBuffers();
   public:
       // set by Opticks::makeEvent OpticksRun::createEvent
       void             setSibling(OpticksEvent* sibling);
       void             setOpticks(Opticks* ok);
       Opticks*         getOpticks() const ;
       OpticksProfile*  getProfile() const ;
       void             setId(int id);
   public:
       OpticksEvent*  getSibling();
       int  getId();
   public:
       bool isNoLoad() const ;
       bool isLoaded() const ;
       bool isStep() const ;
       bool isFlat() const ;
       bool isTorchType();
       bool isMachineryType();

       BTimes* getPrelaunchTimes();
       BTimes* getLaunchTimes();
   public:
       void     setSkipAhead(unsigned skipahead);
       unsigned getSkipAhead() const ;
   public:
       void postPropagateGeant4(); // called following dynamic photon/record/sequence collection
   public:
       // from parameters set in Opticks::makeEvent
       std::string getTimeStamp();
       std::string getCreator();
       char getEntryCode();
       int getDynamic() const ;

       void setTimeStamp(const char* tstamp);
       void setCreator(const char* executable);
       void setEntryCode(char entryCode);
       void setDynamic(int dynamic);
   public:
        const char*    getGeoPath();
       NGeoTestConfig* getTestConfig();
   private:
       std::string getTestCSGPath();
       void        setTestCSGPath(const char* testcsgpath);
       std::string getTestConfigString();
       void        setTestConfigString(const char* testconfig);
   public:
       std::string getNote();
       void        setNote(const char* note);
       void        appendNote(const char* note);
   private:
       void init();
       void indexPhotonsCPU();
       void collectPhotonHitsCPU();
   public:
       static const char* idom_ ;
       static const char* fdom_ ;
       static const char* genstep_ ;
       static const char* nopstep_ ;
       static const char* photon_ ;
       static const char* debug_ ;
       static const char* way_ ;
       static const char* source_ ;
       static const char* record_  ;
       static const char* phosel_ ;
       static const char* recsel_  ;
       static const char* sequence_  ;
       static const char* seed_  ;
       static const char* hit_  ;
       static const char* hiy_  ;
   public:
       bool isIndexed() const ;
       NPY<float>* loadGenstepDerivativeFromFile(const char* stem="track");

       void setGenstepData(NPY<float>* genstep_data_, bool resize_ , bool clone_ );
       void setNopstepData(NPY<float>* nopstep_data_, bool clone_ );
       void setSourceData(NPY<float>* source_data_, bool clone_ );   // from CG4::postpropagate

       void zero();
   public:
       void addBufferControl(const char* name, const char* ctrl_);
       int seedDebugCheck(const char* msg="OpticksEvent::seedDebugCheck");
  private:
       void setBufferControl(NPYBase* data);

   public:
       const char* getPath(const char* xx);  // accepts abbreviated or full constituent names
       BMeta*       getParameters();
   public:
       void makeReport(bool verbose=false);
       void saveReport();
       void loadReport();
       void saveTimes();
       void loadTimes();
   private:
       void saveReport(const char* dir);
       void saveTimes(const char* dir);
   public:
       // G4 related qtys used by cfg4- when OpticksEvent used to store G4 propagations
       void setNumG4Event(unsigned int n);
       void setNumPhotonsPerG4Event(unsigned int n);
       unsigned int getNumG4Event();
       unsigned int getNumPhotonsPerG4Event() const ; 
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

   public:
       //////////  m_domain related ///////////////////////
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);

       // domains used for record compression
       void setSpaceDomain(const glm::vec4& space_domain);
       void setTimeDomain(const glm::vec4& time_domain);
       void setWavelengthDomain(const glm::vec4& wavelength_domain);

   public:
       const glm::vec4& getSpaceDomain() const ;
       const glm::vec4& getTimeDomain() const ;
       const glm::vec4& getWavelengthDomain() const ;

   public:
       // idom persisted 
       unsigned getMaxRec() const ;     // per-photon
       unsigned getMaxBounce() const ;  // per-photon
       unsigned getMaxRng() const ;     // limits number of photons, has been 3M for a long time

       void setMaxRec(unsigned maxrec);         // maximum record slots per photon
       void setMaxBounce(unsigned maxbounce);         // maximum record slots per photon
       void setMaxRng(unsigned maxrng); 

       void dumpDomains(const char* msg="OpticksEvent::dumpDomains");
   private:
       void updateDomainsBuffer();
       void importDomainsBuffer();
       void saveDomains();


   public:
       // huh : these are from m_parameters not m_domain 
       void setRngMax(unsigned int rng_max);  
       unsigned getBounceMax() const ;
       unsigned getRngMax() const ;
   public:
       void save();
   public:
       // used from G4Opticks for the minimal G4 side instrumentation of "1st executable"
       void saveHitData(NPY<float>* ht) const ; 
       void saveHiyData(NPY<float>* hy) const ; 
       void saveSourceData(NPY<float>* so) const ; 
   private:
       void saveHitData() const ; 
       void saveHiyData() const ; 
       void saveSourceData() const ; 
       void saveNopstepData(); 
       void saveGenstepData(); 
       void savePhotonData(); 
       void saveRecordData(); 
       void saveSequenceData(); 
       void saveSeedData(); 
       void saveDebugData(); 
       void saveWayData(); 
       void saveIndex();
       void loadIndex();
       void loadBuffers(bool verbose=true);

   private:
       void importGenstepDataLoaded(NPY<float>* gs);
       void loadBuffersImportSpec(NPYBase* npy, NPYSpec* spec);
   private:
       void createBuffers(); 
       void createSpec(); 
   private:
       void dumpSpec(); 
       void deleteSpec(); 
       void deleteMeta(); 
       void deleteCtrl(); 
       void deleteIndex(); 
       void deleteAttr(); 
       void deleteBuffers(); 
   private:
       void setPhotonData(NPY<float>* photon_data);
       void setSequenceData(NPY<unsigned long long>* history_data);
       void setSeedData(NPY<unsigned>* seed_data);
       void setHitData(NPY<float>* hit_data);
       void setHiyData(NPY<float>* hiy_data);
       void setDebugData(NPY<float>* debug_data);
       void setWayData(NPY<float>* way_data);
       void setRecordData(NPY<short>* record_data);
       void setRecselData(NPY<unsigned char>* recsel_data);
       void setPhoselData(NPY<unsigned char>* phosel_data);
   private:
       void recordDigests();
   public:
       std::string getTagDir(const char* anno=NULL); // anno usually NULL, sometimes the timestamp
       std::string getTagZeroDir(const char* anno=NULL);

       static std::string TagDir(const char* pfx, const char* det, const char* typ, const char* tag, const char* anno=NULL);
   public:
       unsigned long long getSeqHis(unsigned photon_id) const ; 
       unsigned long long getSeqMat(unsigned photon_id) const ; 
       std::string        getSeqHisString(unsigned photon_id) const  ; 

   private:
       void saveParameters();
       void loadParameters();
       void importParameters();

   public:
       bool                 hasGenstepData() const ;
       bool                 hasSourceData() const ;
       bool                 hasPhotonData() const ;
       bool                 hasDebugData() const ;
       bool                 hasWayData() const ;
       bool                 hasRecordData() const ;
   public:
       const glm::vec4&     getGenstepCenterExtent();
   public:
       NPY<float>*          getGenstepData() const ;
       NPY<float>*          getNopstepData() const ;
       NPY<float>*          getPhotonData() const ;
       NPY<float>*          getDebugData() const ;
       NPY<float>*          getWayData() const ;
       NPY<float>*          getSourceData() const ;
       NPY<short>*          getRecordData() const ;
       NPY<unsigned char>*  getPhoselData() const ;
       NPY<unsigned char>*  getRecselData() const ;
       NPY<unsigned long long>*  getSequenceData() const ;
       NPY<unsigned>*          getSeedData() const ;
       NPY<float>*             getHitData() const ;
       NPY<float>*             getHiyData() const ;

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
       //void setHitsNPY(HitsNPY* hit);
       void setBoundariesNPY(BoundariesNPY* bnd);

       RecordsNPY*          getRecordsNPY();  // use OpticksEventInstrument::SetupRecordsNPY 
       PhotonsNPY*          getPhotonsNPY();
       //HitsNPY*             getHitsNPY();
       BoundariesNPY*       getBoundariesNPY();
   public:
       NPY<float>*          getFDomain() const ;
       NPY<int>*            getIDomain() const ;
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
       MultiViewNPY* getHiyAttr();

       ViewNPY* operator [](const char* spec);

   public:
       unsigned int getNumGensteps();
       unsigned int getNumNopsteps();
       unsigned int getNumPhotons() const ;
       unsigned int getNumSource() const ;
       unsigned int getNumRecords() const ;

   public:
       void resizeToZero();  // used by OpticksHub::setupZeroEvent
   private:
       // set by setGenstepData based on summation over Cerenkov/Scintillation photons to generate
       void setNumPhotons(unsigned int num_photons, bool resize=true);
       void resize();
       void setMetadataNum();
   public:
       void Summary(const char* msg="OpticksEvent::Summary");
       std::string  brief() ;  // cannot be const 
       std::string  description(const char* msg="OpticksEvent::description");

       bool         isInterop();
       bool         isCompute();

   private:
       // foreign  
       Opticks*              m_ok ;  
       OpticksProfile*       m_profile ;  
       OpticksMode*          m_mode ; 
   private:
       bool                  m_noload ; 
       bool                  m_loaded ; 
   private:
       // owned : deleteMeta
       BMeta*                m_versions ;
       BMeta*                m_parameters ;
       Report*               m_report ;
       OpticksDomain*        m_domain ; 
       BTimes*               m_prelaunch_times ; 
       BTimes*               m_launch_times ; 
       const char*           m_geopath ; 
   private: 
       // owned : deleteBuffers
       NPY<float>*               m_genstep_data ;
       NPY<float>*               m_nopstep_data ;
       NPY<float>*               m_photon_data ;
       NPY<float>*               m_debug_data ;
       NPY<float>*               m_way_data ;
       NPY<float>*               m_source_data ;
       NPY<short>*               m_record_data ;
       NPY<unsigned char>*       m_phosel_data ;
       NPY<unsigned char>*       m_recsel_data ;
       NPY<unsigned long long>*  m_sequence_data ;
       NPY<unsigned>*            m_seed_data ;
       NPY<float>*               m_hit_data ;
       NPY<float>*               m_hiy_data ;
   private:
       // owned : deleteCtrl
       OpticksBufferControl*  m_photon_ctrl ; 
       OpticksBufferControl*  m_source_ctrl ; 
       OpticksBufferControl*  m_seed_ctrl ; 
   private:
       // weak : constituent of m_genstep_attr 
       ViewNPY*        m_genstep_vpos ;

   private:
       // owned : deleteAttr
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
       MultiViewNPY*   m_hiy_attr  ;

   private:
       // unmanaged
       RecordsNPY*     m_records ; 
       PhotonsNPY*     m_photons ; 
       BoundariesNPY*  m_bnd ; 

   private:
       unsigned int    m_num_gensteps ; 
       unsigned int    m_num_nopsteps ; 
       unsigned int    m_num_photons ; 
       unsigned int    m_num_source ; 

   private:
       // owned : deleteIndex
       Index*          m_seqhis ; 
       Index*          m_seqmat ; 
       Index*          m_bndidx ; 

   private:
       std::vector<std::string>           m_data_names ; 
       std::map<std::string, std::string> m_abbrev ; 

   private:
       // owned: deleteSpec
       NPYSpec* m_fdom_spec ;  
       NPYSpec* m_idom_spec ;  

       NPYSpec* m_genstep_spec ;  
       NPYSpec* m_nopstep_spec ;  
       NPYSpec* m_photon_spec ;  
       NPYSpec* m_debug_spec ;  
       NPYSpec* m_way_spec ;  
       NPYSpec* m_source_spec ;  
       NPYSpec* m_record_spec ;  
       NPYSpec* m_phosel_spec ;  
       NPYSpec* m_recsel_spec ;  
       NPYSpec* m_sequence_spec ;  
       NPYSpec* m_seed_spec ;  
       NPYSpec* m_hit_spec ;  
       NPYSpec* m_hiy_spec ;  

    private:
       OpticksEvent*   m_sibling ;    // weak 

    private:
       // unmanaged, not usually present
       NGeoTestConfig* m_geotestconfig ; 
       const char*     m_fake_nopstep_path ; 
       unsigned        m_skipahead ; 

};

//
// avoiding class members simplifies usage, as full type spec is not needed for pointers : forward declarations sufficient
// for this reason moved glm domain vector members down into OpticksDomain
// this simplifies use with nvcc compiler
// 

#include "OKCORE_TAIL.hh"
  
