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

**/

struct QUDARAP_API QEvent : public SCompProvider
{
    friend struct QEventTest ; 
    friend struct QEvent_setInputPhoton_Test ; 

    static const plog::Severity LEVEL ; 
    static QEvent* INSTANCE ; 
    static QEvent* Get(); 

    sevent* getDevicePtr() const ;

    QEvent(); 

private:
    void init(); 

    // NB members needed on both CPU+GPU or from the QEvent.cu functions 
    // should reside inside the sevent.h instance not up here in QEvent.hh

public:
    SEvt*             sev ;  
private:
    sphoton_selector* selector ; 
    sevent*           evt ; 
    sevent*           d_evt ; 
    NP*               gs ;  
    NP*               input_photon ; 
    NP*               narrow_input_photon ; 
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
    int setGenstep();  // PRIMARY ACTION OF QEvent 

private:
    int  setGenstep(NP* gs);
    void device_alloc_genstep_and_seed(); 
    void setInputPhoton(); 
    void checkInputPhoton() const ; 

    int setGenstep(quad6* gs, unsigned num_gs ); 
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
    // SCompProvider methods
    std::string getMeta() const ;  // returns underlying (SEvt)sev->meta
    NP*      gatherComponent(unsigned comp) const ; 
public:
    NP*      getGenstep() const ; 
    NP*      getInputPhoton() const ; 

    NP*      gatherDomain() const ; 
    NP*      gatherGenstepFromDevice() const ; 
    NP*      gatherSeed() const ; 
    NP*      gatherPhoton() const ; 
    NP*      gatherSimtrace() const ; 
    NP*      gatherSeq() const ;       // seqhis..
    NP*      gatherPrd() const ;  
    NP*      gatherTag() const ;  
    NP*      gatherFlat() const ;  
    NP*      gatherRecord() const ;    // full step records
    NP*      gatherRec() const  ;      // compressed step record
    NP*      gatherHit() const ; 
public:
    // mutating interface. TODO: Suspect only the photon mutating API actually needed for some QSimTest, remove the others
    void     gatherPhoton(       NP* p ) const ;
    void     gatherSimtrace(     NP* t ) const ;
    void     gatherSeq(          NP* seq) const ; 
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

    void checkEvt() ;  // GPU side 

};


