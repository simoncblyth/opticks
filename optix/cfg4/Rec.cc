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
#include "CPropLib.hh"
#include "Rec.hh"

#include "PLOG.hh"


Rec::Rec(CPropLib* clib, OpticksEvent* evt)  
   :
    m_clib(clib),
    m_evt(evt), 
    m_genflag(0),
    m_seqhis(0ull),
    m_seqmat(0ull),
    m_slot(0),
    m_record_max(0),
    m_bounce_max(0),
    m_steps_per_photon(0),
    m_debug(false)
{
   init();
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




void Rec::init()
{
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

void Rec::Clear()
{
    m_states.clear();
    m_seqhis = 0ull ; 
    m_seqmat = 0ull ; 
    m_slot = 0  ; 
}

unsigned int Rec::getNumStates()
{
    return m_states.size();
}

const State* Rec::getState(unsigned int i)
{
    return i < getNumStates() ? m_states[i] : NULL ; 
}


G4OpBoundaryProcessStatus Rec::getBoundaryStatus(unsigned int i)
{
    const State* state = getState(i) ;
    return state ? state->getBoundaryStatus() : Undefined ;
}


Rec::Rec_t Rec::getFlagMaterial(unsigned int& flag, unsigned int& material, unsigned int i, Flag_t type )
{
    // recast of Recorder::RecordStep flag assignment 
    // in after-recording-all-states way instead of live stepping
    // for sanity and checking  

    const State* state = getState(i) ;
    const State* prior = i > 0 ? getState(i-1) : NULL ; 

    const G4StepPoint* pre = state->getPreStepPoint();
    const G4StepPoint* post = state->getPostStepPoint();

    unsigned int preMat  = state->getPreMaterial();
    unsigned int postMat = state->getPostMaterial();

    G4OpBoundaryProcessStatus boundary_status = state->getBoundaryStatus() ;
    G4OpBoundaryProcessStatus prior_boundary_status = prior ? prior->getBoundaryStatus() : Undefined ;

    unsigned int preFlag   = i == 0 ? m_genflag : OpPointFlag(pre,  prior_boundary_status) ; 
    unsigned int postFlag  = OpPointFlag(post, boundary_status) ;

    // winging-it to match Opticks record logic, whilst iterating with pmt_test.py box_test.py 
    // to compare seqmat and seqhis

    bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool preSkip = type == PRE && prior_boundary_status == StepTooSmall ; 

    bool matSwap = boundary_status == StepTooSmall ;  


    if(preSkip) return SKIP_STS ; 

    switch(type)
    {
       case  PRE: 
                  flag = preFlag ; 
                  material = matSwap ? postMat : preMat ;  
                  break;
       case POST: 
                  flag = postFlag ; 
                  material = ( matSwap || postMat == 0 || surfaceAbsorb) ? preMat : postMat ;  
                 // avoid NoMaterial at last step with postMat == 0 causing to use preMat
                 // avoid Bialkali at surfaceAbsorb as Opticks surface treatment never records that 
                 // MAYBE:special case it to set Bialkali, as kinda useful
                  break;
    }

    return OK ; 
}

void Rec::addFlagMaterial(unsigned int flag, unsigned int material)
{
    bool invalid = flag == NAN_ABORT ; 
    bool truncate = m_slot > m_bounce_max  ;  

    if(m_debug)
    LOG(info) << "Rec::addFlagMaterial " 
              << " flag " << std::hex << flag << std::dec
              << " material " << std::hex << material << std::dec
              << " invalid " << invalid 
              << " truncate " << truncate
              ; 

    if(invalid || truncate) return ; 

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;
    unsigned long long shift = slot*4ull ;   
    unsigned long long msk = 0xFull << shift ; 
    unsigned long long his = BBit::ffs(flag) & 0xFull ; 
    unsigned long long mat = material < 0xFull ? material : 0xFull ; 

    m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ;
    m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 

    m_slot += 1 ; 
}

void Rec::sequence()
{
    unsigned int nstep = getNumStates();

    if(m_debug)
    LOG(info) << "Rec::sequence" 
              << " nstep " << nstep 
              ;  

    unsigned int flag ;
    unsigned int material ;
    m_slot = 0 ;
    Rec_t rc ; 

    for(unsigned int i=0 ; i < nstep ; i++)
    {
        rc = getFlagMaterial(flag, material, i, PRE );
        if(rc == OK)
            addFlagMaterial(flag, material) ;
    }

    rc = getFlagMaterial(flag, material, nstep-1, POST );
    if(rc == OK)
        addFlagMaterial(flag, material) ;
}



void Rec::Dump(const char* msg)
{
    unsigned int nstates = m_states.size();
    LOG(info) << msg 
              << " nstates " << nstates 
              ;

    unsigned int preFlag ; 
    unsigned int preMat ; 
    unsigned int postFlag ;
    unsigned int postMat ;

    G4ThreeVector origin ; 

    for(unsigned int i=0 ; i < nstates ; i++)
    {
        const State* state = getState(i) ;
        const State* prior = i > 0 ? getState(i-1) : NULL ; 

        const G4StepPoint* pre  = state->getPreStepPoint() ; 
        const G4StepPoint* post = state->getPostStepPoint() ; 

        if( i == 0)
        {
            const G4ThreeVector& pos = pre->GetPosition();
            origin = pos ; 
        } 

        getFlagMaterial(preFlag,  preMat, i, PRE );
        getFlagMaterial(postFlag, postMat, i, POST );

        unsigned int preMatRaw = state->getPreMaterial();
        unsigned int postMatRaw = state->getPostMaterial();

        const char* preMaterialRaw  = preMatRaw == 0 ? "-" : m_clib->getMaterialName(preMatRaw - 1) ;
        const char* postMaterialRaw = postMatRaw == 0 ? "-" : m_clib->getMaterialName(postMatRaw - 1) ;
   
        
        G4OpBoundaryProcessStatus boundary_status = getBoundaryStatus(i) ;
        G4OpBoundaryProcessStatus prior_boundary_status = i > 0 ? getBoundaryStatus(i-1) : Undefined ;
 
        std::cout << "[" << std::setw(3) << i
                  << "/" << std::setw(3) << nstates
                  << "]"   
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
              << m_clib->MaterialSequence(m_seqmat) 
              << std::endl ;

 


}



