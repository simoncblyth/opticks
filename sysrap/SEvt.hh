#pragma once
/**
SEvt.hh
=========

TODO : lifecycle leak checking 


Q: Where is this instanciated canonically ?
A: So far in G4CXOpticks and in the mains of lower level tests


Note the distinction in the names of accessors:

*gather*
   resource intensive action done few times, that likely allocates memory,
   in the case of sibling class QEvent *gather* mostly includes downloading from device 
   Care is needed in memory management of returned arrays to avoid leaking.  

*get*
   cheap action, done as many times as needed, returning a pointer to 
   already allocating memory that is separately managed, eg within NPFold






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
#include "sphit.h"
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

//struct SCF ; 
struct sphoton_selector ; 
struct sdebug ; 
struct NP ; 
struct NPFold ; 
struct SGeo ; 

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SEvt : public SCompProvider
{
    //static const SCF* CF ;  // TODO: eliminate 
    static constexpr const unsigned UNDEF = ~0u ; 

    int cfgrc ; 
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
    bool              gather_done ; 

    sframe            frame ;

    std::vector<unsigned> comp ; 
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
    static const char*  DEFAULT_RELDIR ; 

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
    static SEvt* Load(); 
    static void Save() ; 
    static void Save(const char* base, const char* reldir ); 
    static void Save(const char* dir); 
    static void SetIndex(int index); 
    static void UnsetIndex(); 
    static int  GetIndex(); 

    static void SetReldir(const char* reldir); 
    static const char* GetReldir(); 

    static int GetNumPhotonFromGenstep(); 
    static int GetNumGenstepFromGenstep(); 
    static int GetNumHit() ; 

    static NP* GatherGenstep(); 
    static NP* GetInputPhoton(); 
    static void SetInputPhoton(NP* ip); 
    static bool HasInputPhoton(); 

 
    SEvt(); 

    const char* getSaveDir() const ; 
    const char* getLoadDir() const ; 
    const char* getSearchCFBase() const ; 


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
    void setFrame(unsigned ins_idx); 


    static quad6 MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr ); 


    void setCompProvider(const SCompProvider* provider); 
    bool isSelfProvider() const ; 
    std::string descProvider() const ; 


    void setNumPhoton(unsigned num_photon); 
    void setNumSimtrace(unsigned num_simtrace);


    void hostside_running_resize(); 

    NP* gatherDomain() const ; 

    void clear() ; 

    void setIndex(int index_) ;  
    void unsetIndex() ;  
    int getIndex() const ; 

    void setReldir(const char* reldir_) ;  
    const char* getReldir() const ; 

    unsigned getNumGenstepFromGenstep() const ; 
    unsigned getNumPhotonFromGenstep() const ; 

    static constexpr const unsigned G4_INDEX_OFFSET = 1000000 ; 
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


    NP* gatherPho0() const ;   // unordered push_back as they come 
    NP* gatherPho() const ;    // resized at genstep and slotted in 
    NP* gatherGS() const ;   // genstep labels from std::vector<sgs>  

    NP* gatherGenstep() const ; 
    NP* gatherPhoton() const ; 
    NP* gatherRecord() const ; 
    NP* gatherRec() const ; 
    NP* gatherSeq() const ; 
    NP* gatherPrd() const ; 
    NP* gatherTag() const ; 
    NP* gatherFlat() const ; 

    NP* gatherSeed() const ; 
    NP* gatherHit() const ; 
    NP* gatherSimtrace() const ; 


    NP* makePhoton() const ; 
    NP* makeRecord() const ; 
    NP* makeRec() const ; 
    NP* makeSeq() const ; 
    NP* makePrd() const ; 
    NP* makeTag() const ; 
    NP* makeFlat() const ; 


    // SCompProvider methods

    std::string getMeta() const ; 
    NP* gatherComponent(unsigned comp) const ; 
    NP* gatherComponent_(unsigned comp) const ; 

    void saveLabels(const char* dir) const ;  // formerly savePho
    void saveFrame(const char* dir_) const ; 

    void saveGenstep(const char* dir) const ; 


    void gather() ;  // with on device running this downloads

    // save methods not const as calls gather
    void save() ; 
    void load() ; 
    void save(const char* base, const char* reldir1, const char* reldir2 ); 
    void save(const char* base, const char* reldir ); 

    const char* getOutputDir(const char* base_=nullptr) const ; 
    void save(const char* dir); 
    void load(const char* dir); 

    std::string desc() const ; 
    std::string descGS() const ; 
    std::string descDir() const ; 
    std::string descFold() const ; 
    std::string descComponent() const ; 
    std::string descComp() const ; 


    const NP* getPhoton() const ; 
    const NP* getHit() const ; 

    unsigned getNumPhoton() const ; 
    unsigned getNumHit() const ; 

    void getPhoton(sphoton& p, unsigned idx) const ; 
    void getHit(   sphoton& p, unsigned idx) const ; 

    void getLocalPhoton(sphoton& p, unsigned idx) const ; 
    void getLocalHit(   sphit& ht, sphoton& p, unsigned idx) const ; 
    void getPhotonFrame( sframe& fr, const sphoton& p ) const ; 

    std::string descNum() const ; 
    std::string descPhoton(unsigned max_print=10) const ; 
    std::string descLocalPhoton(unsigned max_print=10) const ; 
    std::string descFramePhoton(unsigned max_print=10) const ; 
    std::string descFull(unsigned max_print=10) const ; 

    void getFramePhoton(sphoton& p, unsigned idx) const ; 
    void getFrameHit(   sphoton& p, unsigned idx) const ; 
};

