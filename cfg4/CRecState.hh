#pragma once

struct CG4Ctx ; 
#include <string>
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CRecState
{
    const CG4Ctx& _ctx ; 

    unsigned _decrement_request ; 
    unsigned _decrement_denied ; 
    unsigned _topslot_rewrite ; 

    bool     _record_truncate ; 
    bool     _bounce_truncate ; 

    unsigned _slot ; 
    unsigned _step_action ; 

    CRecState(const CG4Ctx& ctx);
    void clear(bool action=true);
    std::string desc() const ; 

    void     decrementSlot();  // NB not just live 
    unsigned constrained_slot();
    void     increment_slot_regardless();
    bool     is_truncate() const ; 



};
 
