#pragma once
/**
SEvt.hh
=========

Q: Where is this instanciated canonically ?


Attempting to do this header only gives duplicate symbol for the SEvt::INSTANCE.
It seems c++17 would allow "static inline SEvt* INSTANCE"  but c++17 
is problematic on laptop, so use separate header and implementation
for simplicity. 

It is possible with c++11 but is non-intuitive

* https://stackoverflow.com/questions/11709859/how-to-have-static-data-members-in-a-header-only-library


Replacing cfg4/CGenstepCollector

HMM: gs vector of sgs provides summary of the full genstep, 
changing the first quad of the genstep to hold this summary info 
would avoid the need for the summary vector and mean the genstep 
index and photon offset info was available on device.

Header of full genstep has plenty of spare bits to squeeze in
index and photon offset in addition to  gentype/trackid/matline/numphotons 

**/

#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "plog/Severity.h"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sgs.h"

struct NP ; 
struct NPFold ; 
struct SCompProvider ; 

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SEvt
{
    std::vector<quad6> genstep ; 
    std::vector<sgs>   gs ; 
    std::vector<spho>  pho0 ;  // unordered push_back as they come 
    std::vector<spho>  pho ;   // spho are label structs holding 4*int 
    std::vector<sphoton> photon ; 
    std::vector<sphoton> record ; 

    //sgs  current_gs ;   NOT USEFUL AS S OFTEN TRUMPS C, NEED TO USE THE GS index in the pho label to get actual sgs genstep label 
    spho current_pho ; 
    sphoton current_photon ; 

    const SCompProvider*  provider ; 
    NPFold*   fold ; 


    static const plog::Severity LEVEL ; 
    static SEvt* INSTANCE ; 
    static SEvt* Get() ; 
    static bool RECORD_PHOTON ; 

    static void Check(); 
    static sgs AddGenstep(const quad6& q); 
    static sgs AddGenstep(const NP* a); 
    static void AddCarrierGenstep(); 
    static void AddTorchGenstep(); 

    static void Clear(); 
    static void Save() ; 
    static void Save(const char* base, const char* reldir ); 
    static void Save(const char* dir); 
    static int GetNumPhoton(); 
    static NP* GetGenstep(); 


    SEvt(); 
    void setCompProvider(const SCompProvider* provider); 

    void clear() ; 
    unsigned getNumGenstep() const ; 
    unsigned getNumPhoton() const ; 
    sgs addGenstep(const quad6& q) ; 
    sgs addGenstep(const NP* a) ; 
    const sgs& get_gs(const spho& sp); // lookup sgs genstep label corresponding to spho photon label  

    void beginPhoton(const spho& sp); 
    void continuePhoton(const spho& sp); 
    void checkPhoton(const spho& sp) const ; 
    void endPhoton(const spho& sp); 


    NP* getPho0() const ;   // unordered push_back as they come 
    NP* getPho() const ;    // resized at genstep and slotted in 
    NP* getGS() const ;   // genstep labels from std::vector<sgs>  
    NP* getPhoton() const ;  // from std::vector<sphoton>  


    void savePho(const char* dir) const ; 

    void saveGenstep(const char* dir) const ; 
    NP* getGenstep() const ; 

    void gather_components() ; 

    static const char* FALLBACK_DIR ; 
    static const char* DefaultDir() ; 

    // save methods not const as calls gather_components 
    void save() ; 
    void save(const char* base, const char* reldir ); 
    void save(const char* dir); 

    std::string desc() const ; 
    std::string descFold() const ; 
    std::string descComponent() const ; 
};



