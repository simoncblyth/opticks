suspect-direct-gensteps-are-stomping-live-ones
===============================================

Gensteps:

1. loaded   <-- this is confusing, but probably harmless
2. collected
3. saved



::

    2020-07-23 02:50:51.679 INFO  [307723] [GGeo::postDirectTranslation@806] ) GGeo::save 
    2020-07-23 02:50:51.679 INFO  [307723] [G4Opticks::translateGeometry@333] ) GGeo::postDirectTranslation 
    2020-07-23 02:50:51.679 FATAL [307723] [G4Opticks::setGeometry@246] ) translateGeometry 
    2020-07-23 02:50:51.679 FATAL [307723] [G4Opticks::setGeometry@260] ( createCollectors 
    2020-07-23 02:50:51.681 FATAL [307723] [G4Opticks::setGeometry@262] ) createCollectors 
    2020-07-23 02:50:51.681 FATAL [307723] [G4Opticks::setGeometry@267] ( OpMgr 
    2020-07-23 02:50:51.681 INFO  [307723] [SLog::SLog@31]  ( OpMgr::OpMgr 
    2020-07-23 02:50:51.686 FATAL [307723] [Opticks::configure@2195]  configured already 
    2020-07-23 02:50:51.686 ERROR [307723] [Opticks::setupTimeDomain@2527]  animtimerange -1.0000,-1.0000,0.0000,0.0000
    2020-07-23 02:50:51.686 INFO  [307723] [Opticks::setupTimeDomain@2538]  cfg.getTimeMaxThumb [--timemaxthumb] 6 cfg.getAnimTimeMax [--animtimemax] -1 cfg.getAnimTimeMax [--animtimemax] -1 speed_of_light (mm/ns) 300 extent (mm) 60000 rule_of_thumb_timemax (ns) 1200 u_timemax 1200 u_animtimemax 1200
    2020-07-23 02:50:51.687 FATAL [307723] [Opticks::setProfileDir@546]  dir /tmp/blyth/opticks/source/evt/g4live/natural
    2020-07-23 02:50:51.739 INFO  [307723] [OpticksGen::init@127] 
    2020-07-23 02:50:51.739 INFO  [307723] [OpticksGen::initFromDirectGensteps@183] /tmp/blyth/opticks/evt/g4live/natural/1/gs.npy
    2020-07-23 02:50:51.739 INFO  [307723] [SLog::SLog@31]  ( OpEngine::OpEngine 
    2020-07-23 02:50:51.740 INFO  [307723] [OContext::Create@235] [
    2020-07-23 02:50:51.740 ERROR [307723] [OContext::SetupOptiXCachePathEnvvar@286] envvar OPTIX_CACHE_PATH not defined setting it internally to /var/tmp/blyth/OptiXCache
    2020-07-23 02:50:51.796 INFO  [307723] [OContext::InitRTX@323]  --rtx 0 setting  OFF
    Missing separate debuginfo for /lib64/libcuda.so
    Try: yum --enablerepo='*debug*' install /usr/lib/debug/.build-id/3e/1e7dd516361182d263c7713bd47eaa498bf0cd.debug
    [New Thread 0x7fff8ffff700 (LWP 309181)]
    2020-07-23 02:50:52.036 INFO  [307723] [OContext::CheckDevices@207] 
    Device 0                        TITAN V ordinal 0 Compute Support: 7 0 Total Memory: 12652838912
    Device 1                      TITAN RTX ordinal 1 Compute Support: 7 5 Total Memory: 25396445184

    2020-07-23 02:50:52.036 INFO  [307723] [OContext::CheckDevices@228]  NULL frame_renderer : compute mode ? 
    [New Thread 0x7fff8d745700 (LWP 309182)]

    ...

    2020-07-23 02:53:14.657 INFO  [307723] [CGenstepCollector::collectScintillationStep@257]  gentype 4 gentype DsG4Scintillation_r3971 pdgCode 11 numPhotons   62 ngs  60 nsc  56 nck   5 nma   0 tot  61
    2020-07-23 02:53:14.657 INFO  [307723] [CGenstepCollector::collectScintillationStep@257]  gentype 4 gentype DsG4Scintillation_r3971 pdgCode 11 numPhotons   16 ngs  61 nsc  57 nck   5 nma   0 tot  62
    2020-07-23 02:53:14.663 INFO  [307723] [CGenstepCollector::collectScintillationStep@257]  gentype 4 gentype DsG4Scintillation_r3971 pdgCode 11 numPhotons  304 ngs  62 nsc  58 nck   5 nma   0 tot  63
    2020-07-23 02:53:14.663 INFO  [307723] [CGenstepCollector::collectScintillationStep@257]  gentype 4 gentype DsG4Scintillation_r3971 pdgCode 11 numPhotons   76 ngs  63 nsc  59 nck   5 nma   0 tot  64
    junoSD_PMT_v2::EndOfEvent opticksMode 1
    [[ junoSD_PMT_v2::EndOfEvent_Opticks  eventID 0 m_opticksMode 1 numGensteps 64 numPhotons 11235
    2020-07-23 02:53:14.692 INFO  [307723] [G4Opticks::propagateOpticalPhotons@450] [[
    2020-07-23 02:53:14.692 INFO  [307723] [G4Opticks::propagateOpticalPhotons@458]  saving gensteps to /tmp/blyth/opticks/evt/g4live/natural/1/gs.npy
    2020-07-23 02:53:14.693 INFO  [307723] [G4Opticks::propagateOpticalPhotons@465]  saving debug gensteps to dbggspath /tmp/blyth/opticks/dbggs/1.npy eventID 0
    2020-07-23 02:53:14.693 INFO  [307723] [OpMgr::propagate@110] 

    [[



