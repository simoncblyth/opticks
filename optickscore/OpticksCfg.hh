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

     const std::string& getG4IniMac();
     const std::string& getG4RunMac();
     const std::string& getG4FinMac();

     const std::string& getG4GunConfig() const ;
     const std::string& getAnaKey() const ;
     const std::string& getAnaKeyArgs() const ;

     const std::string& getTestConfig();
     const std::string& getStateTag();
     const std::string& getMaterialPrefix();

     const std::string& getSnapConfig();
     const std::string& getG4SnapConfig();
     const std::string& getZExplodeConfig();
     const std::string& getMeshVersion();
     const std::string& getRenderMode();
     const std::string& getRenderCmd();
     const std::string& getISlice();
     const std::string& getFSlice();
     const std::string& getPSlice();

     const std::string& getPrintIndex() const ;
     const std::string& getDbgIndex() const ;

     const std::string& getDbgMesh() const ;
     const std::string& getOtherIndex();
     const std::string& getMask() const ;
     const std::string& getX4PolySkip() const ;
     const std::string& getCSGSkipLV() const ;  
     const std::string& getBuilder();
     const std::string& getTraverser();

     const std::string& getSeqMap() const ;
     void setSeqMap(const char* seqmap);    // used from OpticksEventAna

     const std::string& getDbgSeqhis();
     const std::string& getDbgSeqmat();
     const std::string& getLVSDName();
     const std::string& getCathode();
     const std::string& getCerenkovClass();
     const std::string& getScintillationClass();

     const std::string& getFxReConfig();
     const std::string& getFxScConfig();
     const std::string& getFxAbConfig();
     const std::string& getAnalyticPMTSlice();
     const std::string& getAnalyticPMTMedium();

     const std::string& getFlightPathDir();

     float        getEpsilon(); 

     unsigned     getSeed() const ; 
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

     int          getDebugIdx() const ; 
     int          getDbgNode() const ;  
     int          getStack() const ; 

     int          getNumPhotonsPerG4Event(); 
     int          getLoadVerbosity(); 
     int          getImportVerbosity(); 
     int          getMeshVerbosity(); 
     int          getVerbosity(); 
     int          getAnalyticPMTIndex(); 


     const std::string& getSrcGLTFBase();
     const std::string& getSrcGLTFName();
     const std::string& getGLTFConfig();
     int                getGLTF();
     int                getGLTFTarget();
     int                getLayout() const ;

     const std::string& getLODConfig();
     int                getLOD() const ;
     int                getTarget() const ;
     int                getAlignLevel() const ;

     const std::string& getGPUMonPath() const ;

     int                getRunStamp() const ; 
     const std::string& getRunLabel() const ;
     const std::string& getRunFolder() const ;


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
     std::string m_anakeyargs ;

     std::string m_testconfig ;
     std::string m_state_tag ;
     std::string m_materialprefix ;

     std::string m_snapconfig ;
     std::string m_g4snapconfig ;
     std::string m_zexplodeconfig ;
     std::string m_meshversion ;
     std::string m_rendermode ;
     std::string m_rendercmd ;
     std::string m_islice ;
     std::string m_fslice ;
     std::string m_pslice ;
     std::string m_pindex ;
     std::string m_dindex ;
     std::string m_oindex ;
     std::string m_mask ;
     std::string m_x4polyskip ;
     std::string m_csgskiplv ; 
     std::string m_builder ;
     std::string m_traverser  ;
     std::string m_seqmap ;
     std::string m_dbgseqhis ;
     std::string m_dbgseqmat ;
     std::string m_dbgmesh ;

     std::string m_fxreconfig ; 
     std::string m_fxabconfig ; 
     std::string m_fxscconfig ; 
     std::string m_apmtslice ; 

     std::string m_lvsdname ;
     std::string m_cathode ;
     std::string m_cerenkovclass ;
     std::string m_scintillationclass ;

     float       m_epsilon ; 
     unsigned    m_seed ; 
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
     int         m_loadverbosity ; 
     int         m_importverbosity ; 
     int         m_meshverbosity ; 
     int         m_verbosity ; 
     int         m_apmtidx ; 

     std::string m_flightpathdir ; 
     std::string m_apmtmedium ; 
     std::string m_srcgltfbase ; 
     std::string m_srcgltfname ;
     std::string m_gltfconfig ;
     int         m_gltf ;  
     int         m_gltftarget ;  

     int         m_layout ;  

     std::string m_lodconfig ;
     int         m_lod ;  

     int         m_target ;  
     int         m_alignlevel ;  

     const char* m_exename ; 
     std::string m_gpumonpath ;

     int          m_runstamp ; 
     std::string  m_runlabel ; 
     std::string  m_runfolder ; 

};


