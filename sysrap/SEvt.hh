#pragma once
/**
SEvt.hh
=========

Q: Where is this instanciated canonically ?
A: So far in the mains of tests


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
#include "sstate.h"
#include "srec.h"
#include "sseq.h"
#include "stag.h"
#include "sevent.h"
#include "sctx.h"

#include "squad.h"
#include "sframe.h"

#include "sgs.h"
#include "SComp.h"
#include "SRandom.h"

struct SCF ; 
struct sphoton_selector ; 
struct sdebug ; 
struct NP ; 
struct NPFold ; 
struct SGeo ; 

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SEvt : public SCompProvider
{
    static const SCF* CF ;  // TODO: eliminate 

    int index ; 
    const char* reldir ; 
    sphoton_selector* selector ; 
    sevent* evt ; 
    sdebug* dbg ; 
    std::string meta ; 
    NP* input_photon ; 
    NP* input_photon_transformed ; 

    const SRandom*        random ; 
    const SCompProvider*  provider ; 
    NPFold*               fold ; 
    const SGeo*           cf ; 

    bool              hostside_running_resize_done ; // only ever becomes true for non-GPU running 
    sframe            frame ;


    std::vector<quad6> genstep ; 
    std::vector<sgs>   gs ; 
    std::vector<spho>  pho0 ;  // unordered push_back as they come 
    std::vector<spho>  pho ;   // spho are label structs holding 4*int 

    std::vector<int>     slot ; 
    std::vector<sphoton> photon ; 
    std::vector<sphoton> record ; 
    std::vector<srec>    rec ; 
    std::vector<sseq>    seq ; 
    std::vector<quad2>   prd ; 
    std::vector<stag>    tag ; 
    std::vector<sflat>   flat ; 

    // current_* are saved into the vectors on calling SEvt::pointPhoton 
    spho    current_pho = {} ; 
    quad2   current_prd = {} ; 
    sctx    current_ctx = {};  

    static const plog::Severity LEVEL ; 
    static const int PIDX ; 
    static const int GIDX ; 
    static const int MISSING_INDEX ; 

    static SEvt* INSTANCE ; 
    static SEvt* Get() ; 
    static bool RECORDING ; 

    static void Check(); 
    static void AddTag(unsigned stack, float u ); 
    static int  GetTagSlot(); 
    static sgs AddGenstep(const quad6& q); 
    static sgs AddGenstep(const NP* a); 
    static void AddCarrierGenstep(); 
    static void AddTorchGenstep(); 

    static void Clear(); 
    static void Save() ; 
    static void Save(const char* base, const char* reldir ); 
    static void Save(const char* dir); 
    static void SetIndex(int index); 
    static void UnsetIndex(); 
    static int  GetIndex(); 

    static void SetReldir(const char* reldir); 
    static const char* GetReldir(); 

    static int GetNumPhoton(); 
    static int GetNumGenstep(); 
    static NP* GetGenstep(); 
    static NP* GetInputPhoton(); 
    static void SetInputPhoton(NP* ip); 
    static bool HasInputPhoton(); 

 
    SEvt(); 

    const char* getSaveDir() const ; 
    void init(); 

    static const char* INPUT_PHOTON_DIR ; 
    static NP* LoadInputPhoton(); 
    static NP* LoadInputPhoton(const char* ip); 
    void initInputPhoton(); 
    void setInputPhoton(NP* p); 

    NP* getInputPhoton_() const ; 
    NP* getInputPhoton() const ;    // returns input_photon_transformed when exists 
    bool hasInputPhoton() const ; 

    static const bool setFrame_WIDE_INPUT_PHOTON ; 
    void setFrame(const sframe& fr ); 
    void setGeo(const SGeo* cf); 

    static quad6 MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr ); 


    void setCompProvider(const SCompProvider* provider); 
    bool isSelfProvider() const ; 
    std::string descProvider() const ; 


    void setNumPhoton(unsigned num_photon); 
    void setNumSimtrace(unsigned num_simtrace);


    void hostside_running_resize(); 

    NP* getDomain() const ; 

    void clear() ; 
    unsigned getNumGenstep() const ; 

    void setIndex(int index_) ;  
    void unsetIndex() ;  
    int getIndex() const ; 

    void setReldir(const char* reldir_) ;  
    const char* getReldir() const ; 


    unsigned getNumPhoton() const ; 
    unsigned getNumPhotonFromGenstep() const ; 
    sgs addGenstep(const quad6& q) ; 
    sgs addGenstep(const NP* a) ; 

    const sgs& get_gs(const spho& label) const ; // lookup genstep label from photon label  
    unsigned get_genflag(const spho& label) const ; 

    void beginPhoton(const spho& sp); 
    void rjoinPhoton(const spho& sp); 
    unsigned getCurrentPhotonIdx() const ; 

    void rjoinRecordCheck(const sphoton& rj, const sphoton& ph  ) const ; 
    static void ComparePhotonDump(const sphoton& a, const sphoton& b ); 
    void rjoinPhotonCheck(const sphoton& ph) const ; 
    void rjoinSeqCheck(unsigned seq_flag) const ; 

    void checkPhotonLineage(const spho& sp) const ; 
    void pointPhoton(const spho& sp); 
    void finalPhoton(const spho& sp); 

    void addTag(unsigned tag, float u); 
    int getTagSlot() const ; 


    NP* getPho0() const ;   // unordered push_back as they come 
    NP* getPho() const ;    // resized at genstep and slotted in 
    NP* getGS() const ;   // genstep labels from std::vector<sgs>  

    NP* getPhoton() const ; 
    NP* getRecord() const ; 
    NP* getRec() const ; 
    NP* getSeq() const ; 
    NP* getPrd() const ; 
    NP* getTag() const ; 
    NP* getFlat() const ; 

    NP* makePhoton() const ; 
    NP* makeRecord() const ; 
    NP* makeRec() const ; 
    NP* makeSeq() const ; 
    NP* makePrd() const ; 
    NP* makeTag() const ; 
    NP* makeFlat() const ; 


    // SCompProvider methods

    std::string getMeta() const ; 
    NP* getComponent(unsigned comp) const ; 
    NP* getComponent_(unsigned comp) const ; 

    void saveLabels(const char* dir) const ;  // formerly savePho
    void saveFrame(const char* dir_) const ; 

    void saveGenstep(const char* dir) const ; 
    NP* getGenstep() const ; 


    void gather_components() ; 

    static const char* FALLBACK_DIR ; 
    static const char* DefaultDir() ; 

    // save methods not const as calls gather_components 
    void save() ; 
    void load() ; 
    void save(const char* base, const char* reldir1, const char* reldir2 ); 
    void save(const char* base, const char* reldir ); 

    const char* getOutputDir(const char* base_=nullptr) const ; 
    void save(const char* dir); 
    void load(const char* dir); 

    std::string desc() const ; 
    std::string descGS() const ; 
    std::string descFold() const ; 
    std::string descComponent() const ; 


    void getPhoton(sphoton& p, unsigned idx) const ; 
    void getHit(   sphoton& p, unsigned idx) const ; 

    unsigned getNumFoldPhoton() const ; 
    unsigned getNumFoldHit() const ; 

    void getLocalPhoton(sphoton& p, unsigned idx) const ; 
    void getLocalHit(   sphoton& p, unsigned idx) const ; 
    void applyLocalTransform_w2m( sphoton& lp) const ; 
    void getPhotonFrame( sframe& fr, const sphoton& p ) const ; 

    std::string descPhoton(unsigned max_print=10) const ; 
    std::string descLocalPhoton(unsigned max_print=10) const ; 
    std::string descFramePhoton(unsigned max_print=10) const ; 

    void getFramePhoton(sphoton& p, unsigned idx) const ; 
    void getFrameHit(   sphoton& p, unsigned idx) const ; 
};

