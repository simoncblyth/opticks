#pragma once
#include "Opticks.hh"
#include "BCfg.hh"

#include "OKCORE_API_EXPORT.hh"


template <class Listener>
class OKCORE_API OpticksCfg : public BCfg {
  public:
     OpticksCfg(const char* name, Listener* listener, bool live);
  public:
     void dump(const char* msg="OpticksCfg::dump");


     const std::string& getSize();
     const std::string& getPosition();
     const std::string& getDbgCSGPath();
     const std::string& getLogName();
     const std::string& getConfigPath();
     const std::string& getEventTag();
     const std::string& getIntegratedEventTag();
     const std::string& getEventCat();
     const std::string& getLiveLine();

     const std::string& getExportConfig();
     const std::string& getTorchConfig();

     const std::string& getG4GunConfig();
     const std::string& getG4IniMac();
     const std::string& getG4RunMac();
     const std::string& getG4FinMac();
     const std::string& getAnaKey();

     const std::string& getTestConfig();
     const std::string& getStateTag();
     const std::string& getMaterialPrefix();

     const std::string& getSnapConfig();
     const std::string& getZExplodeConfig();
     const std::string& getMeshVersion();
     const std::string& getRenderMode();
     const std::string& getISlice();
     const std::string& getFSlice();
     const std::string& getPSlice();
     const std::string& getPrintIndex();
     const std::string& getDbgIndex();
     const std::string& getDbgMesh() const ;
     const std::string& getOtherIndex();
     const std::string& getBuilder();
     const std::string& getTraverser();
     const std::string& getDbgSeqhis();
     const std::string& getDbgSeqmat();

     const std::string& getFxReConfig();
     const std::string& getFxScConfig();
     const std::string& getFxAbConfig();
     const std::string& getAnalyticPMTSlice();
     const std::string& getAnalyticPMTMedium();

     float        getEpsilon(); 
     int          getRngMax(); 
     int          getBounceMax(); 
     int          getRecordMax(); 
     int          getTimeMax(); 
     int          getAnimTimeMax(); 
     int          getInterpolatedViewPeriod(); 
     int          getOrbitalViewPeriod(); 
     int          getTrackViewPeriod(); 
     int          getAnimatorPeriod(); 
     int          getRepeatIndex(); 
     int          getMultiEvent(); 
     int          getRestrictMesh(); 
     int          getAnalyticMesh(); 
     int          getModulo(); 
     int          getOverride(); 
     int          getDebugIdx(); 
     int          getDbgNode(); 
     int          getStack(); 
     int          getNumPhotonsPerG4Event(); 
     int          getLoaderVerbosity(); 
     int          getMeshVerbosity(); 
     int          getVerbosity(); 
     int          getAnalyticPMTIndex(); 


     const std::string& getGLTFBase();
     const std::string& getGLTFName();
     const std::string& getGLTFConfig();
     int                getGLTF();
     int                getGLTFTarget();

     const std::string& getLODConfig();
     int                getLOD();
     int                getTarget();




private:
     void init();
private:
     Listener*   m_listener ; 
     std::string m_size ;
     std::string m_position ;
     std::string m_dbgcsgpath ;
     std::string m_logname ;
     std::string m_event_cat ;
     std::string m_event_tag ;
     std::string m_integrated_event_tag ;
     std::string m_liveline ;

     std::string m_configpath ;
     std::string m_exportconfig ;
     std::string m_torchconfig ;

     std::string m_g4gunconfig ;
     std::string m_g4inimac ;
     std::string m_g4runmac ;
     std::string m_g4finmac ;
     std::string m_anakey ;

     std::string m_testconfig ;
     std::string m_state_tag ;
     std::string m_materialprefix ;

     std::string m_snapconfig ;
     std::string m_zexplodeconfig ;
     std::string m_meshversion ;
     std::string m_rendermode ;
     std::string m_islice ;
     std::string m_fslice ;
     std::string m_pslice ;
     std::string m_pindex ;
     std::string m_dindex ;
     std::string m_oindex ;
     std::string m_builder ;
     std::string m_traverser  ;
     std::string m_dbgseqhis ;
     std::string m_dbgseqmat ;
     std::string m_dbgmesh ;

     std::string m_fxreconfig ; 
     std::string m_fxabconfig ; 
     std::string m_fxscconfig ; 
     std::string m_apmtslice ; 


     float       m_epsilon ; 
     int         m_rngmax ; 
     int         m_bouncemax ; 
     int         m_recordmax ; 
     int         m_timemax ; 
     int         m_animtimemax ; 
     int         m_animator_period ; 
     int         m_ivperiod ; 
     int         m_ovperiod ; 
     int         m_tvperiod ; 
     int         m_repeatidx ; 
     int         m_multievent ; 
     int         m_restrictmesh; 
     int         m_analyticmesh; 
     int         m_modulo ; 
     int         m_override ; 
     int         m_debugidx ; 
     int         m_dbgnode ; 
     int         m_stack ; 
     int         m_num_photons_per_g4event;
     int         m_loaderverbosity ; 
     int         m_meshverbosity ; 
     int         m_verbosity ; 
     int         m_apmtidx ; 

     std::string m_apmtmedium ; 
     std::string m_gltfbase ; 
     std::string m_gltfname ;
     std::string m_gltfconfig ;
     int         m_gltf ;  
     int         m_gltftarget ;  


     std::string m_lodconfig ;
     int         m_lod ;  

     int         m_target ;  



};


