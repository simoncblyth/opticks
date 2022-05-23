#pragma once

struct qevent ; 
struct quad4 ;
struct sphoton ; 
struct qat4 ; 
struct quad6 ;
struct NP ; 

template <typename T> struct qselector ; 
struct sphoton_selector ; 

template <typename T> struct QBuf ; 

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

/**
QEvent
=======

Canonical *event* instanciated within QSim::QSim 

Unlike typical CPU side event classes with many instances the QEvent/qevent is rather "static"
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

struct QUDARAP_API QEvent
{
    friend struct QEventTest ; 

    static void  CheckGensteps(const NP* gs); 
    static const plog::Severity LEVEL ; 
    static QEvent* INSTANCE ; 
    static QEvent* Get(); 
    static std::string DescGensteps(const NP* gs, int edgeitems=5) ; 
    static std::string DescSeed( const std::vector<int>& seed, int edgeitems ); 

    QEvent(); 
    void init(); 

    // NB members needed on both CPU+GPU or from the QEvent.cu functions 
    // should reside inside the qevent.h instance not up here in QEvent.hh
 
    sphoton_selector* selector ; 

    qevent*      evt ; 
    qevent*      d_evt ; 
    const NP*    gs ;  
    const NP*    p  ; 

    std::string  meta ; 

    int setGenstep();
private:
    int setGenstep(const NP* gs);
    int setGenstep(const quad6* gs, unsigned num_gs ); 
public:
    const NP* getGenstep() const ; 


    bool hasGenstep() const ; 
    bool hasSeed() const ; 
    bool hasPhoton() const ; 
    bool hasRecord() const ; 
    bool hasRec() const ; 
    bool hasSeq() const ; 
    bool hasHit() const ; 

    unsigned count_genstep_photons(); 
    void     fill_seed_buffer(); 
    void     count_genstep_photons_and_fill_seed_buffer(); 

    void     setPhoton( const NP* p );
    void     getPhoton(       NP* p ) const ;
    void     getSeq(          NP* seq) const ; 

    NP*      getPhoton() const ; 
    NP*      getSeq() const ;     // seqhis..
    NP*      getRecord() const ; // full step records
    NP*      getRec() const  ;    // compressed step record
    NP*      getDomain() const ; 

    unsigned getNumHit() const ; 
    NP*      getHit() const ; 
    NP*      getHit_() const ; 


    void save(const char* dir) const ; 
    void save(const char* base, const char* reldir ) const ; 
    void save() const ; 


    void     setNumPhoton(unsigned num_photon) ;  
    void     uploadEvt(); 
    unsigned getNumPhoton() const ;  

    void downloadGenstep( std::vector<quad6>& genstep ); 
    void downloadSeed(    std::vector<int>&   seed ); 
    void downloadPhoton(  std::vector<quad4>& photon ); 
    void downloadRecord(  std::vector<quad4>& record ); 

    void savePhoton( const char* dir, const char* name); 
    void saveGenstep(const char* dir, const char* name); 

    std::string desc() const ; 
    std::string descMax() const ; 
    std::string descNum() const ; 
    std::string descBuf() const ; 

    void saveMeta(   const char* dir, const char* name); 
    void setMeta( const char* meta ); 
    bool hasMeta() const ; 

    void checkEvt() ;  // GPU side 



    qevent* getDevicePtr() const ;
};


