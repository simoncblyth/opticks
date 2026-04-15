#pragma once
/**
U4Scint
========

Before 1100::

    typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;

From 1100::

    typedef G4PhysicsFreeVector G4MaterialPropertyVector;

And from 1100 the "Ordered" methods have been consolidated into G4PhysicsFreeVector
and the class G4PhysicsOrderedFreeVector is dropped.
Try to cope with this without version barnching using edit::

   :%s/PhysicsOrderedFree/MaterialProperty/gc

Maybe will need to add some casts too.

**/

#include <string>
#include <iomanip>

#include "Randomize.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "ssys.h"
#include "NPFold.h"
#include "U4MaterialPropertyVector.h"

struct U4Scint
{
    static constexpr const bool VERBOSE = false ;
    static constexpr const char* PROPS = "SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB" ;
    static U4Scint* Create(const NPFold* materials );

    static const int num_bins = 4096 ;
    static const int hd_factor = 20 ;
    static const bool energy_not_wavelength = false ;


    const NPFold* scint ;
    const char* name ;

    const NP* fast ;
    const NP* slow ;
    const NP* reem ;

    const double epsilon ;
    int mismatch_0 ;
    int mismatch_1 ;
    int mismatch ;

    const G4MaterialPropertyVector* theFastLightVector ;
    const G4MaterialPropertyVector* theSlowLightVector ;
    const G4MaterialPropertyVector* ScintillationIntegral ;

    const NP* icdf ;

    const int num_wlsamp ;
    const NP* wlsamp ;


    U4Scint(const NPFold* fold, const char* name);
    void init();

    std::string desc() const ;
    NPFold* make_fold() const ;
    void save(const char* base, const char* rel=nullptr ) const ;


};



#include "U4ScintCommon.h"


/**
U4Scint::Create
----------------

This is invoked from U4Tree::initScint

**/


inline U4Scint* U4Scint::Create(const NPFold* materials ) // static
{
    std::vector<const NPFold*> subs ;
    std::vector<std::string> names ;
    materials->find_subfold_with_all_keys( subs, names, PROPS );

    int num_subs = subs.size();
    int num_names = names.size() ;
    assert( num_subs == num_names );

    const char* name = num_names > 0 ? names[0].c_str() : nullptr ;
    const NPFold* sub = num_subs > 0 ? subs[0] : nullptr ;
    bool with_scint = name && sub ;

    return with_scint ? new U4Scint(sub, name) : nullptr ;
}


inline U4Scint::U4Scint(const NPFold* scint_, const char* name_)
    :
    scint(scint_),
    name(strdup(name_)),
    fast(scint->get("FASTCOMPONENT")),
    slow(scint->get("SLOWCOMPONENT")),
    reem(scint->get("REEMISSIONPROB")),
    epsilon(0.),
    mismatch_0(NP::DumpCompare<double>(fast, slow, 0, 0, epsilon)),
    mismatch_1(NP::DumpCompare<double>(fast, slow, 1, 1, epsilon)),
    mismatch(mismatch_0+mismatch_1),
    theFastLightVector(U4MaterialPropertyVector::FromArray(fast)),
    theSlowLightVector(U4MaterialPropertyVector::FromArray(slow)),
    ScintillationIntegral(U4ScintCommon::Integral(theFastLightVector)),
    icdf(U4ScintCommon::CreateGeant4InterpolatedInverseCDF(ScintillationIntegral,num_bins,hd_factor,name,energy_not_wavelength)),
    num_wlsamp(ssys::getenvint("U4Scint__num_wlsamp", 0)),
    wlsamp(U4ScintCommon::CreateWavelengthSamples(ScintillationIntegral, num_wlsamp ))
{
    init();
}

inline void U4Scint::init()
{
    if(mismatch > 0 ) std::cerr
        << " mismatch_0 " << mismatch_0
        << " mismatch_1 " << mismatch_1
        << " mismatch " << mismatch
        << std::endl
        ;
    assert( mismatch == 0 );
}



inline std::string U4Scint::desc() const
{
    std::stringstream ss ;
    ss << "U4Scint::desc" << std::endl
       << " name " << name
       << " fast " << ( fast ? fast->sstr() : "-" )
       << " slow " << ( slow ? slow->sstr() : "-" )
       << " reem " << ( reem ? reem->sstr() : "-" )
       << " icdf " << ( icdf ? icdf->sstr() : "-" )
       << " wlsamp " << ( wlsamp ? wlsamp->sstr() : "-" )
       << std::endl
       ;

    std::string str = ss.str();
    return str ;
}

inline NPFold* U4Scint::make_fold() const
{
    NPFold* fold = new NPFold ;
    fold->add("fast", fast) ;
    fold->add("slow", slow) ;
    fold->add("reem", reem) ;
    fold->add("icdf", icdf) ;
    if(wlsamp) fold->add("wlsamp", wlsamp) ;
    return fold ;
}

inline void U4Scint::save(const char* base, const char* rel ) const
{
    NPFold* fold = make_fold();
    fold->save(base, rel);
}


