
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "BBit.hh"
#include "CG4Ctx.hh"
#include "CStep.hh"
#include "CPhoton.hh"

CPhoton::CPhoton(const CG4Ctx& ctx)
    :
    _ctx(ctx) 
{
    clear();
}

void CPhoton::clear()
{
    _badflag = 0 ; 
    _slot = 0 ; 
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


void CPhoton::add(unsigned slot, unsigned flag, unsigned  material)
{

    unsigned long long shift = slot*4ull ;     // 4-bits of shift for each slot 
    unsigned long long  msk = 0xFull << shift ; 

    _slot = slot ; 
    _his = BBit::ffs(flag) & 0xFull ; 
    _mat = material < 0xFull ? material : 0xFull ; 
    _material = material ; 
    _flag = 0x1 << (_his - 1) ; 

    //std::cout << " _flag " << _flag << " _his " << _his << " flag " << flag << std::endl ; 

    assert( _flag == flag ); 

    _mat_prior = ( _seqmat & msk ) >> shift ;
    _his_prior = ( _seqhis & msk ) >> shift ;
    _flag_prior = 0x1 << (_his_prior - 1) ;

    _seqhis =  (_seqhis & (~msk)) | (_his << shift) ; 
    _seqmat =  (_seqmat & (~msk)) | (_mat << shift) ; 
    _mskhis |= flag ;    

}

void CPhoton::scrub_mskhis( unsigned flag )
{
    _mskhis = _mskhis & (~flag)  ;

    //  Decrementing slot and running again does not scrub the AB from the mask
    //  so need to scrub the AB (BULK_ABSORB) when a RE (BULK_REEMIT) from rejoining
    //  occurs.  
    //
    //  This should always be correct as AB is a terminating flag, 
    //  so any REJOINed photon will have an AB in the mask
    //  that needs to be a RE instead.
    //
    //  What about SA/SD ... those should never REjoin ?
}


bool CPhoton::is_rewrite_slot() const 
{
    return _his_prior != 0 && _mat_prior != 0 ;
}

std::string CPhoton::desc() const 
{
    std::stringstream ss ; 
    ss << "CPhoton"
       << " seqhis " << std::setw(20) << std::hex << _seqhis << std::dec 
       << " seqmat " << std::setw(20) << std::hex << _seqmat << std::dec 
       ;

    return ss.str();
}

