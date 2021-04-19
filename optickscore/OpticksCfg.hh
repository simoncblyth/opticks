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

#pragma once
#include "Opticks.hh"
#include "BCfg.hh"

#include "OKCORE_API_EXPORT.hh"

/**
OpticksCfg
===========


::

    OKTest --help | wc -l
         686


TODO:

1. reduce the number of options ? 
 
   * perhaps user/developer modes with user mode presenting just options known to still work and useful for users

2. group the options 

**/

template <class Listener>
class OKCORE_API OpticksCfg : public BCfg {
  public:
     OpticksCfg(const char* name, Listener* listener, bool live);
  public:
     void dump(const char* msg="OpticksCfg::dump");

     static const std::string SIZE_P ; 

     const std::string& getKey();
     const std::string& getCVD();
     const std::string& getSize() const ;
     const std::string& getPosition();
     const std::string& getDbgCSGPath();
     const std::string& getLogName();
     const std::string& getConfigPath();

     const char* getEventTag() const ;
     const char* getIntegratedEventTag() const ;
     const char* getEventCat() const ;
     const char* getEventPfx() const ;

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

     const std::string& getFlightConfig();
     const std::string& getSnapConfig();
     const std::string& getSnapOverridePrefix();

     const std::string& getG4SnapConfig();
     const std::string& getZExplodeConfig();
     const std::string& getMeshVersion();
     const std::string& getRenderMode();
     const std::string& getRenderCmd();
     const std::string& getISlice();
     const std::string& getFSlice();
     const std::string& getPSlice();
     const std::string& getInstanceModulo();

     const std::string& getPrintIndex() const ;

     const std::string& getDbgIndex() const ;
     const std::string& getGenIndex() const ;
     const std::string& getOtherIndex();

     const std::string& getDbgMesh() const ;
     const std::string& getMask() const ;
     const std::string& getDbgHitMask() const ;
     const std::string& getX4PolySkip() const ;
     const std::string& getCSGSkipLV() const ;    // --csgskiplv
     const std::string& getAccel();

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
     float              getFlightPathScale();

     float        getEpsilon() const ; 
     float        getPixelTimeScale() const ; 
     int          getCurFlatSigInt() const ; 
     int          getBoundaryStepSigInt() const ; 


     unsigned     getSkipAheadStep() const ;
     unsigned     getSeed() const ; 
     int          getRTX() const ; 
     int          getRenderLoopLimit() const ; 

     int                getRngMax() const ; 
     unsigned long long getRngSeed() const ; 
     unsigned long long getRngOffset() const ; 

     int          getBounceMax(); 
     int          getRecordMax(); 
     float        getTimeMaxThumb() const ; 
     float        getTimeMax() const ; 
     const std::string& getAnimTimeRange() const ;
     float        getAnimTimeMax() const ;  
     int          getInterpolatedViewPeriod(); 
     int          getOrbitalViewPeriod(); 
     int          getTrackViewPeriod(); 
     int          getAnimatorPeriod(); 
     int          getRepeatIndex(); 
     int          getMultiEvent() const ; 
     const std::string& getEnabledMergedMesh() const ; 
     int          getAnalyticMesh() const ; 
     int          getCameraType() const ; 
     int          getModulo(); 
     int          getGenerateOverride(); 
     int          getPropagateOverride(); 

     int          getDebugIdx() const ; 
     int          getDbgNode() const ;  
     int          getDbgMM() const ;  
     int          getDbgLV() const ;  
     int          getStack() const ; 
     unsigned     getWayMask() const ;   // --waymask 3 
     int          getMaxCallableProgramDepth() const ; 
     int          getMaxTraceDepth() const ; 
     int          getUsageReportLevel() const ; 

     int          getNumPhotonsPerG4Event(); 
     int          getLoadVerbosity(); 
     int          getImportVerbosity(); 
     int          getMeshVerbosity(); 
     int          getVerbosity(); 
     int          getAnalyticPMTIndex(); 


     const std::string& getSrcGLTFBase();
     const std::string& getSrcGLTFName();
     const std::string& getGLTFConfig();
     int                getGLTFTarget();
     int                getLayout() const ;

     const std::string& getLODConfig();
     int                getLOD() const ;
public:
     int                getDomainTarget() const ;
     int                getGenstepTarget() const ;
     int                getTarget() const ;
     const std::string& getTargetPVN() const ;
public:
     int                getAlignLevel() const ;

     const std::string& getGPUMonPath() const ;

     const std::string& getRunComment() const ;
     int                getRunStamp() const ; 
     const std::string& getRunLabel() const ;
     const std::string& getRunFolder() const ;
     const std::string& getDbgGDMLPath() const ;
     const std::string& getDbgGSDir() const ;
     const std::string& getPVName() const ;
     const std::string& getBoundary() const ;
     const std::string& getMaterial() const ;

private:
     void init();
private:
     Listener*   m_listener ; 
     std::string m_key ;
     std::string m_cvd ;
     std::string m_size ;
     std::string m_position ;
     std::string m_dbgcsgpath ;
     std::string m_logname ;
     std::string m_event_cat ;
     std::string m_event_pfx ;
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

     std::string m_flightconfig ;
     std::string m_snapconfig ;
     std::string m_snapoverrideprefix ;
     std::string m_g4snapconfig ;
     std::string m_zexplodeconfig ;
     std::string m_meshversion ;
     std::string m_rendermode ;
     std::string m_rendercmd ;
     std::string m_islice ;
     std::string m_fslice ;
     std::string m_pslice ;
     std::string m_instancemodulo ;
     std::string m_pindex ;
     std::string m_dindex ;
     std::string m_oindex ;
     std::string m_gindex ;
     std::string m_mask ;
     std::string m_dbghitmask ;
     std::string m_x4polyskip ;
     std::string m_csgskiplv ; 
     std::string m_accel ;
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
     float       m_pixeltimescale ; 
     int         m_curflatsigint ; 
     int         m_boundarystepsigint ; 
     unsigned    m_seed ; 
     unsigned    m_skipaheadstep ; 
     int         m_rtx ; 
     int         m_renderlooplimit ; 

     typedef unsigned long long ULL_t ; 
     int         m_rngmax ; 
     ULL_t       m_rngseed ; 
     ULL_t       m_rngoffset ; 
     int         m_rngmaxscale ;  

     int         m_bouncemax ; 
     int         m_recordmax ; 
     float       m_timemaxthumb ; 
     float       m_timemax ; 
     std::string m_animtimerange ;
     float       m_animtimemax ; 
     int         m_animator_period ; 
     int         m_ivperiod ; 
     int         m_ovperiod ; 
     int         m_tvperiod ; 
     int         m_repeatidx ; 
     int         m_multievent ; 
     std::string m_enabledmergedmesh; 
     int         m_analyticmesh; 
     int         m_cameratype; 
     int         m_modulo ; 
     int         m_generateoverride ; 
     int         m_propagateoverride ; 
     int         m_debugidx ; 
     int         m_dbgnode ; 
     int         m_dbgmm ; 
     int         m_dbglv ; 
     int         m_stack ; 
     unsigned    m_waymask ; 
     int         m_maxCallableProgramDepth ; 
     int         m_maxTraceDepth ; 
     int         m_usageReportLevel ; 
     int         m_num_photons_per_g4event;
     int         m_loadverbosity ; 
     int         m_importverbosity ; 
     int         m_meshverbosity ; 
     int         m_verbosity ; 
     int         m_apmtidx ; 

     std::string m_flightpathdir ; 
     float       m_flightpathscale ; 
     std::string m_apmtmedium ; 
     std::string m_srcgltfbase ; 
     std::string m_srcgltfname ;
     std::string m_gltfconfig ;
     int         m_gltftarget ;  

     int         m_layout ;  

     std::string m_lodconfig ;
     int         m_lod ;  

     int         m_domaintarget ;  
     int         m_gensteptarget ;  
     int         m_target ;  
     std::string m_targetpvn ;  
     int         m_alignlevel ;  

     const char* m_exename ; 
     std::string m_gpumonpath ;

     std::string  m_runcomment;
     int          m_runstamp ; 
     std::string  m_runlabel ; 
     std::string  m_runfolder ; 
     std::string  m_dbggdmlpath ; 
     std::string  m_dbggsdir ; 

     std::string  m_pvname ; 
     std::string  m_boundary ; 
     std::string  m_material ; 

};


