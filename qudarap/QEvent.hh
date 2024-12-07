#pragma once

struct sevent ; 
struct quad4 ;
struct sphoton ; 
struct salloc ; 
struct qat4 ; 
struct quad6 ;
struct NP ; 

struct SEvt ; 
struct sphoton_selector ; 


#include <vector>
#include <string>
#include "plog/Severity.h"
#include "SComp.h"
#include "QUDARAP_API_EXPORT.hh"

/**
QEvent
=======

Canonical *event* instanciated within QSim::QSim 

Unlike typical CPU side event classes with many instances the QEvent/sevent is rather "static"
and singular with long lived buffers of defined maximum capacity that get reused for each launch.

* Note that CUDA has no realloc the old OContext::resizeBuffer is an OptiX < 7 extension.
  Hence decided to define maximum buffer sizes as calculated from max number of photons for each launch 
  and arrange that lanches to never exceed those maximums 

* hence all GPU buffers can get allocated once at initialization 
  with the configured maximum sizes and simply get reused from event to event 
  (or more specifically from launch to launch which might not map one-to-one with events)

* this simplifies memory handling as free-ing only needed at termination, so can get away with 
  not doing it  

* so "resizing" just becomes changing a CPU/GPU side constant eg *num_photons* 
* just need to ensure that launches are always arranged to be below the max

* how to decide the maximums ? depends on available VRAM and also should be user configurable
  within some range 


QEvent::setGenstep is the primary method for lifecycle understanding
----------------------------------------------------------------------- 

When Opticks is integrated with a Geant4 based detector simulation framework 
this primary QEvent::setGenstep method is invoked with a stack like the below, 
where the upper part depends on details of how Opticks is integrated with the simulation framework::
 
    "G4VSensitiveDetector::EndOfEvent(G4HCofThisEvent* HCE)" (stub overridded by the below) 
    junoSD_PMT_v2::EndOfEvent(G4HCofThisEvent* HCE)
    junoSD_PMT_v2_Opticks::EndOfEvent(G4HCofThisEvent*, int eventID )
    junoSD_PMT_v2_Opticks::EndOfEvent_Simulate(int eventID )
    G4CXOpticks::simulate(int eventID, bool reset_ )
    QSim::simulate(int eventID, bool reset_)
    QEvent::setGenstep


**/

struct QUDARAP_API QEvent : public SCompProvider
{
    friend struct QEventTest ; 
    friend struct QEvent_setInputPhoton_Test ; 

    static constexpr const char* QEvent__LIFECYCLE = "QEvent__LIFECYCLE" ; 
    static bool LIFECYCLE ; 

    static const plog::Severity LEVEL ; 
    static QEvent* INSTANCE ; 
    static QEvent* Get(); 
    static const bool SEvt_NPFold_VERBOSE ; 
    static std::string Desc(); 


    sevent* getDevicePtr() const ;

    QEvent(); 

private:
    void init(); 
    void init_SEvt(); 

    // NB members needed on both CPU+GPU or from the QEvent.cu functions 
    // should reside inside the sevent.h instance not up here in QEvent.hh

public:
    SEvt*             sev ;  
private:
    sphoton_selector* selector ; 
    sevent*           evt ; 
    sevent*           d_evt ; 
    const NP*         gs ;  
    NP*               input_photon ; 
public:
    int               upload_count ; 

    /**
    std::string       meta ; 
    Q: IS THIS meta NEEDED ? SEvt HAS meta TOO ? 
    A: YES, for now. The metadata gets collected in SEvt::gather_components 
       via SCompProvider method QEvent::getMeta (OR SEvt::getMeta) 

    A2: Dont need the meta, need the method that access the underlying SEvt.  
    **/
public:
    // PRIMARY ACTION OF QEvent : genstep uploading 
    int setGenstep();  
    int setGenstepUpload_NP(const NP* gs);
    int setGenstepUpload_NP(const NP* gs,  int gs_start, int gs_stop );
private:

    int setGenstepUpload(const quad6* qq0, int num_gs ); 
    int setGenstepUpload(const quad6* qq0, int gs_start, int gs_stop ); 

    void device_alloc_genstep_and_seed(); 
    void setInputPhoton(); 
    void checkInputPhoton() const ; 

    //int setGenstep(quad6* gs, unsigned num_gs ); 
    unsigned count_genstep_photons(); 
    void     fill_seed_buffer(); 
    void     count_genstep_photons_and_fill_seed_buffer(); 

public:
    // who uses these ? TODO: switch to comp based 
    bool hasGenstep() const ; 
    bool hasSeed() const ; 
    bool hasPhoton() const ; 
    bool hasRecord() const ; 
    bool hasRec() const ; 
    bool hasSeq() const ; 
    bool hasPrd() const ; 
    bool hasTag() const ; 
    bool hasFlat() const ; 
    bool hasHit() const ; 
    bool hasSimtrace() const ; 
public:
    static constexpr const char* TYPENAME = "QEvent" ; 
    // SCompProvider methods
    std::string getMeta() const ;  // returns underlying (SEvt)sev->meta
    const char* getTypeName() const ; 
    NP*      gatherComponent(unsigned comp) const ; 
public:
    // [ expedient getters : despite these coming from SEvt 
    NP*      getGenstep() const ; 
    NP*      getInputPhoton() const ; 
    // ]

    NP*      gatherPhoton() const ; 
    NP*      gatherHit() const ; 

#ifndef PRODUCTION
    NP*      gatherSeed() const ; 
    NP*      gatherDomain() const ; 
    NP*      gatherGenstepFromDevice() const ; 
    void     gatherSimtrace(     NP* t ) const ;
    NP*      gatherSimtrace() const ; 
    void     gatherSeq(          NP* seq) const ; 
    NP*      gatherSeq() const ;       // seqhis..
    NP*      gatherPrd() const ;  
    NP*      gatherFlat() const ;  
    NP*      gatherRecord() const ;    // full step records
    NP*      gatherTag() const ;  
    NP*      gatherRec() const  ;      // compressed step record
#endif

public:
    // mutating interface. TODO: Suspect only the photon mutating API actually needed for some QSimTest, remove the others
    void     gatherPhoton(       NP* p ) const ;

private:
    NP*      gatherComponent_(unsigned comp) const ; 
    NP*      gatherHit_() const ; 
public:
    unsigned getNumHit() const ; 
private:
    void     setNumPhoton(unsigned num_photon) ;  
    void     setNumSimtrace(unsigned num_simtrace) ;  
    void     device_alloc_photon(); 
    void     device_alloc_simtrace(); 
    static void SetAllocMeta(salloc* alloc, const sevent* evt); 
    void     uploadEvt(); 
public:
    unsigned getNumPhoton() const ;  
    unsigned getNumSimtrace() const ;  
public:
    std::string desc() const ; 
    std::string desc_alloc() const ; 

    void checkEvt() ;  // GPU side 

};


