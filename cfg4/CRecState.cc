
#include <sstream>
#include "CG4Ctx.hh"
#include "CRecState.hh"
#include "CAction.hh"
#include "PLOG.hh"

CRecState::CRecState(const CG4Ctx& ctx)
    :
    _ctx(ctx)
{
    clear();
}

void CRecState::clear()
{
    _decrement_request = 0 ;
    _decrement_denied = 0 ;
    _topslot_rewrite =  0 ;

    _record_truncate = false ;
    _bounce_truncate = false ;

    _slot = 0 ;

    _step_action = 0 ;
}


std::string CRecState::desc() const 
{
    std::stringstream ss ; 
    ss << "CRecState" ;

    return ss.str();
}


void CRecState::decrementSlot()  // NB not just live 
{
    _decrement_request += 1 ; 

    if(_slot == 0 )
    {
        _decrement_denied += 1 ; 
        _step_action |= CAction::DECREMENT_DENIED ; 

        LOG(warning) << "CRecState::decrementSlot DENIED "
                     << " slot " << _slot 
                     << " record_truncate " << _record_truncate 
                     << " bounce_truncate " << _bounce_truncate 
                     << " decrement_denied " << _decrement_denied
                     << " decrement_request " << _decrement_request
                      ;  
    }
    else
    {
        _slot -= 1 ; 
    }
}


/**
CRecState::constrained_slot
-------------------------------

Canonically invoked by CPhoton::add

Returns _slot if within constraints, otherwise returns top slot.
So the returned slot is constrained to inclusive range (0,_steps_per_photon-1) 

Side effects:

* _record_truncate is set when the slot returned is the top one

**/

unsigned CRecState::constrained_slot()
{
    unsigned slot =  _slot < _ctx._steps_per_photon ? _slot : _ctx._steps_per_photon - 1 ;
 
    _record_truncate = slot == _ctx._steps_per_photon - 1 ;  // hmm not exactly truncate, just top slot 

    if(_record_truncate) _step_action |= CAction::RECORD_TRUNCATE ; 

    return slot ; 
} 

/**
CRecState::increment_slot_regardless
--------------------------------------

Canonically invoked by CPhoton::increment_slot which is done by CWriter::writeStepPoint

* _slot is incremented regardless of truncation, only the local (and returned) *slot* is constrained to recording range

TODO: can constrained_slot and 


**/

void CRecState::increment_slot_regardless()
{
    _slot += 1 ;    

    _bounce_truncate = _slot > _ctx._bounce_max  ;   

    // huh ? why not updating _record_truncate ? ... better for those to be dynamic 

    if(_bounce_truncate) _step_action |= CAction::BOUNCE_TRUNCATE ; 
}


/**
CRecState::is_truncate
-----------------------

true for bounce or record truncate

**/

bool CRecState::is_truncate() const 
{
    return _bounce_truncate || _record_truncate  ;
}


