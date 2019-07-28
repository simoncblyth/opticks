
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "BBit.hh"

#include "OpticksFlags.hh"

#include "CG4Ctx.hh"
#include "CRecState.hh"

#include "CAction.hh"
#include "CStep.hh"
#include "CPhoton.hh"

#include "PLOG.hh"


CPhoton::CPhoton(const CG4Ctx& ctx, CRecState& state)
    :
    _ctx(ctx),
    _state(state)
{
    clear();
}


void CPhoton::clear()
{
    _badflag = 0 ; 
    _slot_constrained = 0 ; 
    _material = 0 ; 

    _c4.u = 0 ;   // union { u, i, f, char4, uchar4 }

    _c4.uchar_.x = _ctx._step ? CStep::PreQuadrant(_ctx._step) : 0u ; // initial quadrant 
    _c4.uchar_.y = 2u ; 
    _c4.uchar_.z = 3u ; 
    _c4.uchar_.w = 4u ; 


    _seqhis = 0 ; 
    _seqmat = 0 ; 
    _mskhis = 0 ; 

    _his = 0 ; 
    _mat = 0 ; 
    _flag = 0 ; 

    _his_prior = 0 ; 
    _mat_prior = 0 ; 
    _flag_prior = 0 ; 
}

/**
CPhoton::add
---------------

Inserts the argument flag and material into the 
seqhis and seqmat nibbles of the current constrained slot. 

* sets _flag _material 
* updates priors to hold what will be overwritten
* DOES NOT update the slot 
* via CRecState::constrained_slot sets _state._record_truncate when at top slot 

**/

void CPhoton::add(unsigned flag, unsigned  material)
{
    if(flag == 0 ) 
    {
        LOG(fatal) << " _badflag " << _badflag ; 
        _badflag += 1 ; 
        _state._step_action |= CAction::ZERO_FLAG ; 
        assert(0); // check boundary_status and WhateverG4OpBoundaryProcess setup : usual cause of badflags, eg using default when should be custom
    }

    unsigned slot = _state.constrained_slot(); 
    unsigned long long shift = slot*4ull ;     // 4-bits of shift for each slot 
    unsigned long long  msk = 0xFull << shift ; 

    _slot_constrained = slot ; 

    _his = BBit::ffs(flag) & 0xFull ; 

    //  BBit::ffs result is a 1-based bit index of least significant set bit 
    //  so anding with 0xF although looking like a bug, as the result of ffs is not a nibble, 
    //  is actually providing a warning as are constructing seqhis from nibbles : 
    //  this is showing that NATURAL is too big to fit in its nibble   
    //
    //  BUT NATURAL is an input flag meaning either CERENKOV or SCINTILATION, thus
    //  it should not be here at the level of a photon.  It needs to be set 
    //  at genstep level to the appropriate thing. 
    //
    //  See notes/issues/ckm-okg4-CPhoton-add-flag-mismatch-NATURAL-bit-index-too-big-for-nibble.rst      
    //

    _flag = 0x1 << (_his - 1) ; 

    bool flag_match = _flag == flag  ; 
    if(!flag_match)
       LOG(fatal) << "flag mismatch "
                  << " MAYBE TOO BIG TO FIT IN THE NIBBLE " 
                  << " _flag " << _flag 
                  << " _his " << _his 
                  << " flag " << flag 
                  ; 
     assert( flag_match ); 

    _mat = material < 0xFull ? material : 0xFull ; 
    _material = material ; 


    _mat_prior = ( _seqmat & msk ) >> shift ;
    _his_prior = ( _seqhis & msk ) >> shift ;
    _flag_prior = _his_prior > 0 ? 0x1 << (_his_prior - 1) : 0 ;  // turning the 1-based bit index into a flag 

    _seqhis =  (_seqhis & (~msk)) | (_his << shift) ; 
    _seqmat =  (_seqmat & (~msk)) | (_mat << shift) ; 
    _mskhis |= flag ;    

    if(flag == BULK_REEMIT) scrub_mskhis(BULK_ABSORB)  ;
}

/**
CPhoton::increment_slot
-----------------------

Canonically invoked by CWriter::writeStepPoint

**/

void CPhoton::increment_slot()
{   
    _state.increment_slot_regardless();
}


/**
CPhoton::scrub_mskhis
------------------------

Decrementing slot and running again does not scrub the AB from the mask
so need to scrub the AB (BULK_ABSORB) when a RE (BULK_REEMIT) from rejoining
occurs.  

This should always be correct as AB is a terminating flag, 
so any REJOINed photon will have an AB in the mask
that needs to be a RE instead.

What about SA/SD ... those should never REjoin ?

**/

void CPhoton::scrub_mskhis( unsigned flag )
{
    _mskhis = _mskhis & (~flag)  ;
}


/**
CPhoton::is_rewrite_slot
--------------------------

Prior non-zeros in current slot of seqmat/seqhis indicate are overwriting nibbles

**/

bool CPhoton::is_rewrite_slot() const  
{
    return _his_prior != 0 && _mat_prior != 0 ;
}


/**
CPhoton::is_flag_done
------------------------

Returns true when _flag is terminal.

**/

bool CPhoton::is_flag_done() const 
{
    bool flag_done = ( _flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT | MISS)) != 0 ;
    return flag_done ;  
}


/**
CPhoton::is_done
---------------------

Returns true when last added _flag is terminal  

**/


bool CPhoton::is_done() const 
{
    return _state.is_truncate() || is_flag_done() ;   
}


/**
CPhoton::is_hard_truncate
----------------------------

Canonically invoked from CWriter::writeStepPoint

* hard truncation means that are prevented from rewriting the top slot.

* hmm: because of the is_rewrite_slot check should not return true
  when the top slot is first written, only subsequently  

    
Formerly at truncation, rerunning overwrote "the top slot" 
of seqhis,seqmat bitfields (which are persisted in photon buffer)
and the record buffer. 
As that is different from Opticks behaviour for the record buffer
where truncation is truncation, a HARD_TRUNCATION has been adopted.

* notes/issues/geant4_opticks_integration/tconcentric_pflags_mismatch_from_truncation_handling.rst

**/

bool CPhoton::is_hard_truncate()
{
    bool hard_truncate = false ; 

    if(_state._record_truncate && is_rewrite_slot() )  // trying to overwrite top slot 
    {
        _state._topslot_rewrite += 1 ; 

        // allowing a single AB->RE rewrite is closer to Opticks

        if(_state._topslot_rewrite == 1 && _flag == BULK_REEMIT && _flag_prior  == BULK_ABSORB)
        {
            _state._step_action |= CAction::TOPSLOT_REWRITE ; 
        }
        else
        {
            _state._step_action |= CAction::HARD_TRUNCATE ; 
            hard_truncate = true ; 
        }
    }
    return hard_truncate ; 
}


std::string CPhoton::desc() const 
{
    std::stringstream ss ; 
    ss << "CPhoton"
       << " slot_constrained " << _slot_constrained
       << " seqhis " << std::setw(20) << std::hex << _seqhis << std::dec 
       << " seqmat " << std::setw(20) << std::hex << _seqmat << std::dec 
       << " is_flag_done " << ( is_flag_done() ? "Y" : "N" )
       << " is_done " << ( is_done() ? "Y" : "N" )
       ;

    return ss.str();
}


std::string CPhoton::brief() const 
{
    std::stringstream ss ; 
    ss 
       << " seqhis " << std::setw(20) << std::hex << _seqhis << std::dec 
       << " seqmat " << std::setw(20) << std::hex << _seqmat << std::dec 
       ;

    return ss.str();
}

