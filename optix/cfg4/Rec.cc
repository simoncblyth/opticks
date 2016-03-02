#include "Rec.hh"
#include "Format.hh"
#include "State.hh"
#include "OpStatus.hh"
#include "Opticks.hh"
#include "CPropLib.hh"

// opticks-
#include "OpticksPhoton.h"

// npy-
#include "NLog.hpp"

// g4-
#include "G4Step.hh"


void Rec::init()
{
    m_genflag = TORCH ; 
}

void Rec::add(const State* state)
{
    m_states.push_back(state);
}

void Rec::Clear()
{
    m_states.clear();
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
    return state->getBoundaryStatus();
}


unsigned int Rec::getFlag(unsigned int i, Flag_t type )
{
    // recast of Recorder::RecordStep flag assignment 
    // in after-recording-all-states way instead of live stepping
    // for sanity and checking  

    const State* state = getState(i) ;
    const State* prior = i > 0 ? getState(i-1) : NULL ; 

    const G4StepPoint* pre = state->getPreStepPoint();
    const G4StepPoint* post = state->getPostStepPoint();

    G4OpBoundaryProcessStatus boundary_status = state->getBoundaryStatus() ;
    G4OpBoundaryProcessStatus prior_boundary_status = prior ? prior->getBoundaryStatus() : Undefined ;

    unsigned int preFlag   = i == 0 ? m_genflag : OpPointFlag(pre,  prior_boundary_status) ; 
    unsigned int postFlag  = OpPointFlag(post, boundary_status) ;
   
    unsigned int flag ; 
    switch(type)
    {
       case  PRE: flag = preFlag ; break;
       case POST: flag = postFlag ; break;
    }
    return flag ; 
}



void Rec::Dump(const char* msg)
{

    unsigned int nstates = m_states.size();
    LOG(info) << msg 
              << " nstates " << nstates 
              ;

    for(unsigned int i=0 ; i < nstates ; i++)
    {
        const State* state = m_states[i] ;

        const G4Step* step = state->m_step ;
        const G4StepPoint* pre  = step->GetPreStepPoint() ; 
        const G4StepPoint* post = step->GetPostStepPoint() ; 

        unsigned int preFlag = getPreFlag(i);
        unsigned int postFlag = getPostFlag(i);

        const char* preMaterial  = state->m_premat == 0 ? "-" : m_clib->getMaterialName(state->m_premat - 1) ;
        const char* postMaterial = state->m_postmat == 0 ? "-" : m_clib->getMaterialName(state->m_postmat - 1) ;
    
        std::cout << "[" << std::setw(3) << i
                  << "/" << std::setw(3) << nstates
                  << "]"   
                  << std::endl
                  << ::Format("stepStatus", OpStepString(pre->GetStepStatus()), OpStepString(post->GetStepStatus()) )
                  << std::endl
                  << ::Format("flag", Opticks::Flag(preFlag), Opticks::Flag(postFlag) )
                  << std::endl
                  << " bs " 
                  << OpBoundaryAbbrevString(state->m_boundary_status)
                  << std::endl
                  << " " << std::setw(15) << preMaterial
                  << "/" << std::setw(15) << postMaterial
                  << std::endl 
                  << ::Format(state->m_step, "rec state" )
                  << std::endl ; 

    }

}







