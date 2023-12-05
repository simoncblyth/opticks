#pragma once
/**
SEvt.hh
=========

TODO : prune useless/misleading statics, 
       now that commonly have two active SEvt instances 
       it is better to only use statics where really 
       need to message all instances. 


TODO : lifecycle leak checking 

Q: Where is this instanciated canonically ?
A: So far in G4CXOpticks::setGeometry and in the mains of lower level tests

gather vs get
-----------------

Note the distinction in the names of accessors:

*gather*
   resource intensive action done few times, that likely allocates memory,
   in the case of sibling class QEvent *gather* mostly includes downloading from device 
   Care is needed in memory management of returned arrays to avoid leaking.  

*get*
   cheap action, done as many times as needed, returning a pointer to 
   already allocated memory that is separately managed, eg within NPFold


header-only INSTANCE problem 
------------------------------

Attempting to do this header only gives duplicate symbol for the SEvt::INSTANCE.
It seems c++17 would allow "static inline SEvt* INSTANCE"  but c++17 
is problematic on laptop, so use separate header and implementation
for simplicity. 

It is possible with c++11 but is non-intuitive

* https://stackoverflow.com/questions/11709859/how-to-have-static-data-members-in-a-header-only-library

HMM : Alt possibility for merging sgs label into genstep header
------------------------------------------------------------------

gs vector of sgs provides summary of the full genstep, 
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
#include "sprof.h"

#include "squad.h"
#include "sframe.h"

#include "sgs.h"
#include "SComp.h"
#include "SRandom.h"

struct sphoton_selector ; 
struct sdebug ; 
struct NP ; 
struct NPFold ; 
struct SGeo ; 
struct S4RandomArray ;  
struct stimer ; 

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SEvt : public SCompProvider
{
    friend struct SEvtTest ; 

    static constexpr const char* SEvt__LIFECYCLE = "SEvt__LIFECYCLE" ; 
    static bool LIFECYCLE ; 

    static constexpr const char* SEvt__CLEAR_SIGINT = "SEvt__CLEAR_SIGINT" ; 
    static bool CLEAR_SIGINT ; 

    enum { SEvt__SEvt, 
           SEvt__init, 
           SEvt__beginOfEvent, 
           SEvt__endOfEvent,
           SEvt__gather,
           SEvt__clear_output,
           SEvt__clear_genstep,
           SEvt__OTHER
           } ; 

    static constexpr const char* SEvt__SEvt_ = "SEvt__SEvt" ;  
    static constexpr const char* SEvt__init_ = "SEvt__init" ;  
    static constexpr const char* SEvt__beginOfEvent_ = "SEvt__beginOfEvent" ;  
    static constexpr const char* SEvt__endOfEvent_ = "SEvt__endOfEvent" ;  
    static constexpr const char* SEvt__gather_ = "SEvt__gather" ;  
    static constexpr const char* SEvt__clear_output_ = "SEvt__clear_output" ;  
    static constexpr const char* SEvt__OTHER_ = "SEvt__OTHER" ;  

    const char* descStage() const ; 
    void setStage(int stage_); 
    int  getStage() const ; 


    int cfgrc ; 

    int index ; 
    int instance ; 
    int stage ; 

    sprof p_SEvt__beginOfEvent_0 ; 
    sprof p_SEvt__beginOfEvent_1 ; 
    sprof p_SEvt__endOfEvent_0 ;
    //sprof p_SEvt__endOfEvent_1 ;

    uint64_t t_BeginOfEvent ; 
#ifndef PRODUCTION
    uint64_t t_setGenstep_0 ; 
    uint64_t t_setGenstep_1 ; 
    uint64_t t_setGenstep_2 ; 
    uint64_t t_setGenstep_3 ; 
    uint64_t t_setGenstep_4 ; 
    uint64_t t_setGenstep_5 ; 
    uint64_t t_setGenstep_6 ; 
    uint64_t t_setGenstep_7 ; 
    uint64_t t_setGenstep_8 ; 
#endif
    uint64_t t_PreLaunch ; 
    uint64_t t_PostLaunch ; 
    uint64_t t_EndOfEvent ; 

    uint64_t t_PenultimatePoint ; 
    uint64_t t_LastPoint ; 


    double   t_Launch ; 

    sphoton_selector* selector ; 
    sevent* evt ; 
    sdebug* dbg ; 
    std::string meta ; 

    NP* input_genstep ; 
    NP* input_photon ; 
    NP* input_photon_transformed ; 
    NP* g4state ;    // populated by U4Engine::SaveStatus

    const SRandom*        random ; 
    S4RandomArray*        random_array ; 
    // random_array
    //      low-dep random consumption debug, usage needs Geant4  
    //      but its useful to have somewhere to hang it  

    const SCompProvider*  provider ; 
    NPFold*               fold ; 
    const SGeo*           cf ; 

    bool              hostside_running_resize_done ; // only ever becomes true for non-GPU running 
    bool              gather_done ; 
    bool              is_loaded ; 
    bool              is_loadfail ; 

    sframe            frame ;

    // comp vectors are populated from SEventConfig in SEvt::init
    std::vector<unsigned> gather_comp ;
    std::vector<unsigned> save_comp ;

    unsigned           numgenstep_collected ;   // updated by addGenstep
    unsigned           numphoton_collected ;    // updated by addGenstep 
    unsigned           numphoton_genstep_max ;  // maximum photons in a genstep since last SEvt::clear_genstep_vector
    int                clear_genstep_vector_count ; 
    int                clear_output_vector_count ; 

    // moved here from G4CXOpticks, updated in gather 
    unsigned gather_total ; 
    unsigned genstep_total ; 
    unsigned photon_total ; 
    unsigned hit_total ; 


    // [--- these vectors are cleared by SEvt::clear_genstep_vector
    std::vector<quad6>   genstep ; 
    std::vector<sgs>     gs ; 
    // ] 

    // [--- these vectors are cleared by SEvt::clear_output_vector
    std::vector<spho>    pho ;   // spho are label structs holding 4*int  
    std::vector<int>     slot ; 
    std::vector<sphoton> photon ; 
    std::vector<sphoton> record ; 
    std::vector<srec>    rec ; 
    std::vector<sseq>    seq ; 
    std::vector<quad2>   prd ; 
    std::vector<stag>    tag ; 
    std::vector<sflat>   flat ; 
    std::vector<quad4>   simtrace ; 
    std::vector<quad4>   aux ; 
    std::vector<quad6>   sup ; 
    // ]---- these vectors are cleared by SEvt::clear_output_vector


    // current_* are saved into the vectors on calling SEvt::pointPhoton 
    spho    current_pho = {} ; 
    quad2   current_prd = {} ; 
    sctx    current_ctx = {};  


    static constexpr const int M = 1000000 ; 
    static constexpr const unsigned UNDEF = ~0u ; 
    static bool IsDefined(unsigned val); 

    static stimer* TIMER ; 
    static void   TimerStart(); 
    static double TimerDone(); 
    static uint64_t TimerStartCount(); 
    static std::string TimerDesc(); 


    static NP* Init_RUN_META(); 
    static NP* RUN_META ; 
    static std::string* RunMetaString();  


    static NP* UU ;  
    static NP* UU_BURN ;  
    static const plog::Severity LEVEL ; 
    static const int GIDX ; 
    static const int PIDX ; 
    static const int MISSING_INDEX ; 
    static const int MISSING_INSTANCE ; 
    static const int DEBUG_CLEAR ; 

    //static SEvt* INSTANCE ; 
    enum { MAX_INSTANCE = 2 } ;  
    enum { EGPU, ECPU }; 
    static std::array<SEvt*, MAX_INSTANCE> INSTANCES ; 
    static std::string DescINSTANCE(); 

private:

    SEvt(); 
    void init(); 
public:
    void setFoldVerbose(bool v); 

    static const char* GetSaveDir(int idx) ; 
    const char* getSaveDir() const ; 
    const char* getLoadDir() const ; 
    int getTotalItems() const ; 

    static constexpr const char* SearchCFBase_RELF = "CSGFoundry/solid.npy" ; 
    const char* getSearchCFBase() const ; 


    static const char* INPUT_GENSTEP_DIR ; 
    static const char* INPUT_PHOTON_DIR ; 
    static const char* ResolveInputArray(const char* spec, const char* dir) ; 
    static NP* LoadInputArray(const char* path) ; 

    static NP* LoadInputGenstep(); 
    static NP* LoadInputGenstep(const char* spec); 

    static NP* LoadInputPhoton(); 
    static NP* LoadInputPhoton(const char* spec); 



    void initInputGenstep(); 
    void setInputGenstep(NP* g); 
    NP* getInputGenstep() const ; 
    bool hasInputGenstep() const ; 


    void initInputPhoton(); 
    void setInputPhoton(NP* p); 

    NP* getInputPhoton_() const ; 
    bool hasInputPhoton() const ; 
    NP* getInputPhoton() const ;    // returns input_photon_transformed when exists 
    bool hasInputPhotonTransformed() const ; 



    NP* gatherInputGenstep() const ;   // returns a copy of the input genstep array 
    NP* gatherInputPhoton() const ;   // returns a copy of the input photon array 







    void initG4State() ; 
    NP* makeG4State() const ; 
    void setG4State(NP* state) ; 
    NP* gatherG4State() const ;
    const NP* getG4State() const ;


    void setFrame(const sframe& fr ); 
    void setFramePlaceholder(); 

    static const bool transformInputPhoton_WIDE ; 
    void transformInputPhoton(); 

    void addInputGenstep(); 
    void assertZeroGenstep();


    const char* getFrameId() const ; 
    const NP*   getFrameArray() const ; 
    static const char* GetFrameId(int idx) ;
    static const NP*   GetFrameArray(int idx) ;
 
    void setFrame_HostsideSimtrace() ; 
    void setGeo(const SGeo* cf); 
    void setFrame(unsigned ins_idx);  // requires setGeo to access the frame from SGeo

    //// below decl order matches impl order : KEEP IT THAT WAY 


    static SEvt* CreateSimtraceEvent(); 


    void setCompProvider(const SCompProvider* provider); 
    bool isSelfProvider() const ; 
    std::string descProvider() const ; 


    NP* gatherDomain() const ; 

    static int  Count() ; 
    static SEvt* Get_EGPU() ; 
    static SEvt* Get_ECPU() ; 
    static SEvt* Get(int idx) ; 
    static void Set(int idx, SEvt* inst); 

    static SEvt* Create_EGPU() ; 
    static SEvt* Create_ECPU() ; 
    static SEvt* Create(int idx) ; 
    static SEvt* CreateOrReuse(int idx) ; 
    static SEvt* HighLevelCreateOrReuse(int idx) ; 
    static SEvt* HighLevelCreate(int idx); // Create with bells-and-whistles needed by eg u4/tests/U4SimulateTest.cc

    static void CreateOrReuse(); 
    static void SetFrame(const sframe& fr ); 

    bool isEGPU() const ; 
    bool isECPU() const ; 
    bool isFirstEvt() const ; 
    bool isLastEvt() const ; 

    SEvt* getSibling() const ; 


    static bool Exists(int idx); 
    static bool Exists_ECPU(); 
    static bool Exists_EGPU(); 
    static void Check(int idx);

#ifndef PRODUCTION 
    static void AddTag(int idx, unsigned stack, float u ); 
    static int  GetTagSlot(int idx); 
#endif

    static sgs AddGenstep(const quad6& q); 
    static sgs AddGenstep(const NP* a); 
    static void AddCarrierGenstep(); 
    static void AddTorchGenstep(); 
    void addTorchGenstep(); 

    static SEvt* LoadAbsolute(const char* dir); 
    static SEvt* LoadRelative(const char* rel=nullptr);  // formerly Load

    static void ClearOutput(); 
    static void ClearGenstep(); 
    static void Save() ; 
    //static void SaveExtra(const char* name, const NP* a) ; 

    static void Save(const char* bas, const char* rel ); 
    static void Save(const char* dir); 
    static bool HaveDistinctOutputDirs(); 


    static void SaveGenstepLabels(const char* dir, const char* name="gsl.npy"); 

    static void BeginOfRun(); 
    static void EndOfRun(); 


    template<typename T>
    static void SetRunMeta(const char* k, T v ); 

    static void SetRunMetaString(const char* k, const char* v ); 

    static void SetRunProf(const char* k, const sprof& v); 
    static void SetRunProf(const char* k);  // NOW 
    void setRunProf_Annotated(const char* hdr) const  ; 


    static void SaveRunMeta(const char* base=nullptr ); 




    void setMetaString(const char* k, const char* v); 
    void setMetaProf(  const char* k, const sprof& v); 
    void setMetaProf(  const char* k) ; 

    template<typename T>
    void setMeta( const char* k, T v ); 


    void beginOfEvent(int eventID); 
    void endOfEvent(int eventID); 
    void endMeta(); 


    static bool IndexPermitted(int index);   // index is 1-based 
    static int  GetIndex(int idx); 
    static S4RandomArray* GetRandomArray(int idx); 

    static const char*  DEFAULT_RELDIR ; 
    static const char* RELDIR ; 
    static void SetReldir(const char* reldir); 
    static const char* GetReldir(); 


    static int GetNumPhotonCollected(int idx); 
    static int GetNumPhotonGenstepMax(int idx); 
    static int GetNumPhotonFromGenstep(int idx); 
    static int GetNumGenstepFromGenstep(int idx); 
    static int GetNumHit(int idx) ; 
    static int GetNumHit_EGPU() ; 
    static int GetNumHit_ECPU() ; 


    static NP* GatherGenstep(int idx); 
    static NP* GetInputPhoton(int idx); 
    static void SetInputPhoton(NP* ip); 
    static bool HasInputPhoton(int idx); 

    static bool HasInputPhoton(); 
    static NP* GetInputPhoton(); 
    static std::string DescHasInputPhoton(); 

private:
    void clear_genstep_vector() ; 
    void clear_output_vector() ; 
public:
    void clear_genstep() ; 
    void clear_output() ; 

    void setIndex(int index_arg) ;  
    void endIndex(int index_arg) ;  
    int  getIndexArg() const ; 
    int  getIndex() const ; 
    int  getIndexPresentation() const ; 
    std::string descIndex() const ; 

    void incrementIndex() ;  
    void unsetIndex() ;  

    void setInstance(int instance); 
    int getInstance() const ; 


    unsigned getNumGenstepFromGenstep() const ; // number of collected gensteps from size of collected gensteps vector
    unsigned getNumPhotonFromGenstep() const ;  // total photons since last clear from looping over collected gensteps

    unsigned getNumGenstepCollected() const ;   // total collected genstep since last clear
    unsigned getNumPhotonCollected() const ;    // total collected photons since last clear
    unsigned getNumPhotonGenstepMax() const ;   // max photon in genstep since last clear

    static constexpr const unsigned G4_INDEX_OFFSET = 1000000 ; 
    sgs addGenstep(const NP* a) ; 
    sgs addGenstep(const quad6& q) ; 

    void setNumPhoton(unsigned num_photon); 
    void setNumSimtrace(unsigned num_simtrace);
    void hostside_running_resize(); 
    void hostside_running_resize_(); 

    const sgs& get_gs(const spho& label) const ; // lookup genstep label from photon label  
    unsigned get_genflag(const spho& label) const ; 

    void beginPhoton(const spho& sp); 
    unsigned getCurrentPhotonIdx() const ; 

    void resumePhoton(const spho& sp);  // FastSim->SlowSim resume 
    void rjoin_resumePhoton(const spho& label); // reemission rjoin AND FastSim->SlowSim resume 
    void rjoinPhoton(const spho& sp);   // reemission rjoin


    void rjoinRecordCheck(const sphoton& rj, const sphoton& ph  ) const ; 
    static void ComparePhotonDump(const sphoton& a, const sphoton& b ); 
    void rjoinPhotonCheck(const sphoton& ph) const ; 
    void rjoinSeqCheck(unsigned seq_flag) const ; 

    void pointPhoton(const spho& sp); 

    static bool PIDX_ENABLED ;

#ifndef PRODUCTION
    void addTag(unsigned tag, float u); 
    int getTagSlot() const ; 
#endif

    void finalPhoton(const spho& sp); 

    static void AddProcessHitsStamp(int idx, int p); 
    void addProcessHitsStamp(int p) ; 

    void checkPhotonLineage(const spho& sp) const ; 
    
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
///////// below methods handle gathering arrays and persisting, not array content //////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

    NP* gatherPho0() const ;   // unordered push_back as they come 
    NP* gatherPho() const ;    // resized at genstep and slotted in 
    NP* gatherGS() const ;   // genstep labels from std::vector<sgs>  

    NP*    gatherGenstep() const ;  // from genstep vector
    quad6* getGenstepVecData() const ; 
    int    getGenstepVecSize() const ; 


    bool haveGenstepVec() const ; 


    NP* gatherPhoton() const ; 
    NP* gatherRecord() const ; 
    NP* gatherRec() const ; 
    NP* gatherAux() const ; 
    NP* gatherSup() const ; 
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
    NP* makeAux() const ; 
    NP* makeSup() const ; 
    NP* makeSeq() const ; 
    NP* makePrd() const ; 
    NP* makeTag() const ; 
    NP* makeFlat() const ; 
    NP* makeSimtrace() const ; 


    static constexpr const char* TYPENAME = "SEvt" ; 

    //[ SCompProvider methods
    std::string getMeta() const ; 
    const char* getTypeName() const ; 
    NP* gatherComponent(unsigned comp) const ; 
    //] SCompProvider methods


    NP* gatherComponent_(unsigned comp) const ; 

    void saveGenstep(const char* dir) const ; 
    void saveGenstepLabels(const char* dir, const char* name="gsl.npy") const ; 

    std::string descGS() const ; 
    std::string descDir() const ; 
    std::string descFold() const ; 
    static std::string Brief() ; 
    std::string brief() const ; 
    std::string id() const ; 

    std::string desc() const ; 
    std::string descDbg() const ; 

    void gather_components(); 
    void gather_metadata(); 
    void gather() ;           // with on device running this downloads

    // add extra metadata arrays to be saved within SEvt fold 
    void add_array( const char* k, const NP* a ); 
    void addEventConfigArray() ; 


    // save methods not const as call gather
    void save() ; 
    void saveExtra( const char* name, const NP* a ) const ; 

    int  load() ; 

    void onload(); 


    void save(const char* base, const char* reldir ); 
    void save(const char* base, const char* reldir1, const char* reldir2 ); 

    bool hasIndex() const ; 
    bool hasInstance() const ; 

    const char* getOutputDir_OLD(const char* base_=nullptr) const ; 
    const char* getOutputDir(const char* base_=nullptr) const ; 

    char getInstancePrefix() const ; 
    std::string getIndexString_(const char* hdr) const ; 
    const char* getIndexString(const char* hdr) const ; 


    static const char* RunDir( const char* base_=nullptr ); 
    static const char* DefaultDir() ; 

    std::string descSaveDir(const char* dir_) const ; 

    int  load(const char* dir); 
    int  loadfold( const char* dir ); 

    void save(const char* dir); 
    void saveExtra(const char* dir_, const char* name, const NP* a ) const ; 
    void saveFrame(const char* dir_) const ; 

    std::string descComponent() const ; 
    std::string descComp() const ; 
    std::string descVec() const ; 

    const NP* getGenstep() const ; 
    const NP* getPhoton() const ; 
    const NP* getHit() const ; 
    const NP* getAux() const ; 
    const NP* getSup() const ; 
    const NP* getPho() const ; 
    const NP* getGS() const ; 

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


    std::string descInputGenstep() const ; 
    std::string descInputPhoton() const ; 

    static std::string DescInputGenstep(int idx) ; 
    static std::string DescInputPhoton(int idx) ; 

    std::string descFull(unsigned max_print=10) const ; 

    void getFramePhoton(sphoton& p, unsigned idx) const ; 
    void getFrameHit(   sphoton& p, unsigned idx) const ; 
    void applyFrameTransform(sphoton& lp) const ; 

    static NP* CountNibbles( const NP* seq ); 

};

