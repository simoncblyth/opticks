#pragma once

#include <string>
#include "CRecorder.h"
#include "CFG4_API_EXPORT.hh"


struct CFG4_API CPhoton
{
    unsigned _slot ; 
    unsigned _material ; 
    uifchar4   _c4 ; 

    unsigned long long _seqhis ; 
    unsigned long long _seqmat ; 
    unsigned long long _mskhis ; 

    unsigned long long _his ; 
    unsigned long long _mat ; 
    unsigned long long _flag ; 

    unsigned long long _his_prior ; 
    unsigned long long _mat_prior ; 
    unsigned long long _flag_prior ; 

    CPhoton();
    void clear();
    void add(unsigned slot, unsigned flag, unsigned  material);
    bool is_rewrite_slot() const  ;

    std::string desc() const ; 


};
 
