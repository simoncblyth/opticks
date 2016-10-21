#include "CFG4_BODY.hh"
// npy-
#include "BBit.hh"

// okc-
#include "Opticks.hh"
#include "OpticksPhoton.h"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"


// g4-
#include "G4Step.hh"

// cfg4-
#include "Format.hh"
#include "State.hh"
#include "OpStatus.hh"
#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "Rec.hh"

#include "PLOG.hh"


const char* Rec::OK_ = "OK" ; 
const char* Rec::SKIP_STS_ = "SKIP_STS" ; 
const char* Rec::SKIP_REJOIN_ = "SKIP_REJOIN" ; 
const char* Rec::Label(Rec_t r)
{
    const char*  l = NULL ; 
    switch(r)
    {
        case OK         : l = OK_          ;break;
        case SKIP_STS   : l = SKIP_STS_    ;break;
        case SKIP_REJOIN: l = SKIP_REJOIN_ ;break;
    }
    return l ; 
}
 

Rec::Rec(Opticks* ok, CGeometry* geometry, bool dynamic)  
   :
    m_ok(ok), 
    m_geometry(geometry),
    m_material_bridge(NULL),
    m_dynamic(dynamic),
    m_evt(NULL), 
    m_genflag(0),
    m_seqhis(0ull),
    m_seqmat(0ull),
    m_slot(0),
    m_record_max(0),
    m_bounce_max(0),
    m_steps_per_photon(0),
//    m_rejoin_count(0),
    m_bail_count(0),
    m_debug(false)
{
}

void Rec::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
}


void Rec::setDebug(bool debug)
{
    m_debug = debug ; 
}



unsigned long long Rec::getSeqHis()
{
    return m_seqhis ; 
}
unsigned long long Rec::getSeqMat()
{
    return m_seqmat ; 
}


void Rec::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
    assert(m_evt && m_evt->isG4());
}
void Rec::initEvent(OpticksEvent* evt)
{
    setEvent(evt);

    m_record_max = m_evt->getNumPhotons(); 
    m_bounce_max = m_evt->getBounceMax();
    m_steps_per_photon = m_evt->getMaxRec() ;    

    const char* typ = m_evt->getTyp();
    m_genflag = OpticksFlags::SourceCode(typ);

    assert( m_genflag == TORCH || m_genflag == G4GUN );
}


void Rec::add(const State* state)
{
    m_states.push_back(state);
}
//void Rec::pop()
//{
//    m_states.pop_back();
//}
//void Rec::notifyRejoin()
//{
//    m_rejoin_count += 1 ; 
//}



void Rec::Clear()
{
    m_states.clear();
    m_seqhis = 0ull ; 
    m_seqmat = 0ull ; 
    m_slot = 0  ; 
   // m_rejoin_count = 0 ; 
}

unsigned int Rec::getNumStates()
{
    return m_states.size();
}

const State* Rec::getState(unsigned int i)
{
    return i < getNumStates() ? m_states[i] : NULL ; 
}


#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus Rec::getBoundaryStatus(unsigned int i)
#else
G4OpBoundaryProcessStatus Rec::getBoundaryStatus(unsigned int i)
#endif
{
    const State* state = getState(i) ;
    return state ? state->getBoundaryStatus() : Undefined ;
}


CStage::CStage_t Rec::getStage(unsigned int i)
{
    const State* state = getState(i) ;
    return state ? state->getStage() : CStage::UNKNOWN ;
}

double Rec::getPreGlobalTime(unsigned i)
{
    const State* state = getState(i) ;
    const G4StepPoint* pre = state ? state->getPreStepPoint() : NULL ; 
    return pre ? pre->GetGlobalTime() : -1 ;
}

double Rec::getPostGlobalTime(unsigned i)
{
    const State* state = getState(i) ;
    const G4StepPoint* post  = state ? state->getPostStepPoint() : NULL ; 
    return post ? post->GetGlobalTime() : -1 ;
}



Rec::Rec_t Rec::getFlagMaterialStageDone(unsigned int& flag, unsigned int& material,  CStage::CStage_t& stage, bool& done, unsigned int i, Flag_t type )
{
    // recast of Recorder::RecordStep flag assignment 
    // in after-recording-all-states way instead of live stepping
    // for sanity and checking  

    const State* prior = i > 0 ? getState(i-1) : NULL ; 
    const State* state = getState(i) ;
    const State* next  = getState(i+1) ; 

    const G4StepPoint* pre = state->getPreStepPoint();
    const G4StepPoint* post = state->getPostStepPoint();

    unsigned int preMat  = state->getPreMaterial();
    unsigned int postMat = state->getPostMaterial();
#ifdef USE_CUSTOM_BOUNDARY
    DsG4OpBoundaryProcessStatus boundary_status = state->getBoundaryStatus() ;
    DsG4OpBoundaryProcessStatus prior_boundary_status = prior ? prior->getBoundaryStatus() : Undefined ;
#else
    G4OpBoundaryProcessStatus boundary_status = state->getBoundaryStatus() ;
    G4OpBoundaryProcessStatus prior_boundary_status = prior ? prior->getBoundaryStatus() : Undefined ;
#endif

    CStage::CStage_t current_stage = state->getStage() ; 
    CStage::CStage_t next_stage = next ? next->getStage() : CStage::UNKNOWN ; 

    bool current_done = state->getDone();
    //bool next_done = next ? next->getDone() : false ; 

    // zero shunting to include m_genflag in pole position
    // means must use prior_boundary_status and prior_stage for all the PRE to avoid skipping getState(0)
    unsigned int preFlag = i == 0 ?  m_genflag : OpPointFlag(pre,  prior_boundary_status, current_stage) ; 
    unsigned int postFlag =                      OpPointFlag(post, boundary_status,       current_stage ) ;

    // NB nothing fundamental here : just winging it in adhoc attempt to duplicate CRecorder and Opticks logic
    bool reemSkip = current_stage == CStage::REJOIN && next_stage == CStage::REJOIN ; 
    if(reemSkip) return SKIP_REJOIN ; 


    bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool preSkip = type == PRE && prior_boundary_status == StepTooSmall ; 

    bool matSwap = boundary_status == StepTooSmall ;  

    if(preSkip) return SKIP_STS ; 



    switch(type)
    {
       case  PRE: 
                  flag = preFlag ; 
                  material = matSwap ? postMat : preMat ;  
                  stage = current_stage ; 
                  done = current_done ;
                  break;
       case POST: 
                  flag = postFlag ; 
                  stage = current_stage ; 
                  done = current_done ;

                  //  Spring 2016
                  // material = ( matSwap || postMat == 0 || surfaceAbsorb) ? preMat : postMat ;  
                  //
                  // avoid NoMaterial at last step with postMat == 0 causing to use preMat
                  // avoid Bialkali at surfaceAbsorb as Opticks surface treatment never records that 
                  // MAYBE:special case it to set Bialkali, as kinda useful
                  //
                  // Oct 2016:  have changed oxrap/cu/generate.cu to record m2 material in seqmat now for SA and SD
                  //            so try to do the same here
                  //
                  material = ( matSwap || postMat == 0 ) ? preMat : postMat ;  
                  material = ( surfaceAbsorb ) ? postMat : material ; 

                  break;
    }

    return OK ; 
}

void Rec::addFlagMaterial(unsigned int flag, unsigned int material, CStage::CStage_t stage)
{
    bool invalid = flag == NAN_ABORT ; 
    bool truncate = m_slot > m_bounce_max ; 
    bool truncate_not_rejoin = truncate && stage != CStage::REJOIN ; 
    //  _not_rejoin special case needed to match the result of Recorder::decrementSlot 
    //  allowing the changing of a BULK_ABSORB into a BULK_REEMIT 

    bool bail = invalid || truncate_not_rejoin ;   

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;

    if(m_debug)
    LOG(info) << "Rec::addFlagMaterial " 
              << " m_slot " << m_slot 
              << " slot " << slot 
              << " flag " << std::hex << flag << std::dec
              << " flagffs " << std::hex << BBit::ffs(flag) << std::dec
              << " flagffs& " << std::hex << (BBit::ffs(flag) & 0xFull) << std::dec
              << " material " << std::hex << material << std::dec
              << " invalid " << ( invalid ? "Y" : "N" )
              << " truncate " << ( truncate ? "Y" : "N" )
              << " bail " << ( bail ? "Y" : "N" )
              ; 


    // CRecorder::decrementSlot allows rewriting of topslot in
    // special case of REJOIN

    if(bail)
    {
        LOG(debug) << "NAN_ABORT or bounce truncate bail out " 
                     << " bail " << ( bail ? "Y" : "N" )
                     << " truncate_not_rejoin " << ( truncate_not_rejoin ? "Y" : "N" )
                     << " truncate " << ( truncate ? "Y" : "N" )
                     << " invalid " << ( invalid ? "Y" : "N" )
                      ; 
        m_bail_count += 1 ; 
    }
    else
    {
        unsigned long long shift = slot*4ull ;   
        unsigned long long msk = 0xFull << shift ; 
        unsigned long long his = BBit::ffs(flag) & 0xFull ; 
        unsigned long long mat = material < 0xFull ? material : 0xFull ; 

        m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ;
        m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 
        m_slot += 1 ; 
    }

}

void Rec::sequence()
{
    // NB externally changing slot will 
    // not work like it does with CRecorder
    // as the cycle is controlled entirely
    // here from the saved states
    //
    // Note that do not need to do anything special
    // to rejoin reemission as the BULK_ABSORB that 
    // goes into POST gets ignored when there is a subsequent state
    // with BULK_REEMIT in PRE
    //
    // Presumably this is relying on the 2ndaries from reemission 
    // being propagated immediately following the BULK_ABSORB
    //

    unsigned int nstate = getNumStates();

    if(m_debug)
    LOG(info) << "Rec::sequence" 
              << " nstate" << nstate 
              ;  

    unsigned preFlag ;
    unsigned postFlag ;
    unsigned material ;
    CStage::CStage_t stage ; 
    bool done = false ; 

    m_slot = 0 ;
    Rec_t rc ; 


   // add all PRE, until lastPost when add POST

    for(unsigned i=0 ; i < nstate; i++)
    {
        rc = getFlagMaterialStageDone(preFlag, material, stage, done, i, PRE );
        if(m_debug) LOG(info) << "PRE" << std::setw(3) << i << " " << Label(rc) << std::setw(10) << getPreGlobalTime(i)  ;
        if(rc == OK)
            addFlagMaterial(preFlag, material, stage) ;

    }

    rc = getFlagMaterialStageDone(postFlag, material, stage, done, nstate-1, POST );
    // hmm lastPost kinda assumes complete propagation with no truncation 
    bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
    if(m_debug) LOG(info) << "PST" << std::setw(3) << nstate-1 << " " << Label(rc) << std::setw(10) << getPostGlobalTime(nstate-1) << " lastPost " << ( lastPost ? "Y" : "N" ) ;

    if(rc == OK && lastPost )
        addFlagMaterial(postFlag, material, stage) ;

}



void Rec::Dump(const char* msg)
{
    unsigned int nstates = m_states.size();
    LOG(info) << msg 
              << " nstates " << nstates 
              << " bail_count " << m_bail_count 
              ;

    unsigned int preFlag ; 
    unsigned int preMat ; 
    unsigned int postFlag ;
    unsigned int postMat ;
    CStage::CStage_t preStage ;
    CStage::CStage_t postStage ;
    bool preDone ; 
    bool postDone ; 


    G4ThreeVector origin ; 

    for(unsigned int i=0 ; i < nstates ; i++)
    {
        const State* state = getState(i) ;

        const G4StepPoint* pre  = state->getPreStepPoint() ; 
        const G4StepPoint* post = state->getPostStepPoint() ; 

        if( i == 0)
        {
            const G4ThreeVector& pos = pre->GetPosition();
            origin = pos ; 
        } 

        getFlagMaterialStageDone(preFlag,  preMat, preStage, preDone,   i, PRE );
        getFlagMaterialStageDone(postFlag, postMat,postStage,postDone,  i, POST );

        unsigned int preMatRaw = state->getPreMaterial();
        unsigned int postMatRaw = state->getPostMaterial();

        const char* preMaterialRaw  = preMatRaw == 0 ? "-" : m_material_bridge->getMaterialName(preMatRaw - 1) ;
        const char* postMaterialRaw = postMatRaw == 0 ? "-" : m_material_bridge->getMaterialName(postMatRaw - 1) ;
   
       
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = getBoundaryStatus(i) ;
        DsG4OpBoundaryProcessStatus prior_boundary_status = i > 0 ? getBoundaryStatus(i-1) : Undefined ;
#else 
        G4OpBoundaryProcessStatus boundary_status = getBoundaryStatus(i) ;
        G4OpBoundaryProcessStatus prior_boundary_status = i > 0 ? getBoundaryStatus(i-1) : Undefined ;
#endif
 
        std::cout << "[" << std::setw(3) << i
                  << "/" << std::setw(3) << nstates
                  << "]"   
                  << std::endl
                  << ::Format("done",   (preDone ? "preDone" : "") , (postDone ? "postDone" : "" ) )
                   << std::endl
                  << ::Format("stepStage",  CStage::Label(preStage), CStage::Label(postStage) )
                  << std::endl
                  << ::Format("stepStatus", OpStepString(pre->GetStepStatus()), OpStepString(post->GetStepStatus()) )
                  << std::endl
                  << ::Format("flag", OpticksFlags::Flag(preFlag), OpticksFlags::Flag(postFlag) )
                  << std::endl
                  << ::Format("bs pri/cur", OpBoundaryAbbrevString(prior_boundary_status),OpBoundaryAbbrevString(boundary_status))
                  << std::endl
                  << ::Format("material",  preMaterialRaw, postMaterialRaw )
                  << std::endl 
                  << ::Format(state->getStep(),origin,  "rec state" )
                  << std::endl ; 

    }

    std::cout << "(rec)FlagSequence "
              << OpticksFlags::FlagSequence(m_seqhis) 
              << std::endl ;

    std::cout << "(rec)MaterialSequence "
              << m_material_bridge->MaterialSequence(m_seqmat) 
              << std::endl ;

}



