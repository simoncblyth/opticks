#pragma once
/**
U4ScintThree.h : three species LAB/PPO/bisMSB LS model
========================================================


ABSLENGTH.npy
FASTCOMPONENT.npy SLOWCOMPONENT.npy
REEMISSIONPROB.npy

bisMSBABSLENGTH.npy
bisMSBCOMPONENT.npy
bisMSBREEMISSIONPROB.npy
bisMSBTIMECONSTANT.npy

PPOABSLENGTH.npy
PPOCOMPONENT.npy
PPOREEMISSIONPROB.npy
PPOTIMECONSTANT.npy



AlphaCONSTANT.npy
GammaCONSTANT.npy
NeutronCONSTANT.npy
OpticalCONSTANT.npy

GROUPVEL.npy
RAYLEIGH.npy
RINDEX.npy




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
#include "U4ScintCommon.h"

struct U4ScintThree
{
    static constexpr const bool VERBOSE = false ;
    static constexpr const char* ABSLENGTH_PROPS = "ABSLENGTH,PPOABSLENGTH,bisMSBABSLENGTH" ;
    static constexpr const char* REEMISSIONPROB_PROPS = "REEMISSIONPROB,PPOREEMISSIONPROB,bisMSBREEMISSIONPROB" ;
    static constexpr const char* COMPONENT_PROPS = "FASTCOMPONENT,SLOWCOMPONENT,PPOCOMPONENT,bisMSBCOMPONENT" ;


    static const int num_bins = 4096 ;
    static const int hd_factor = 20 ;
    static const bool energy_not_wavelength = false ;



    int   num_wlsamp ;
    const NPFold* scint ;
    const char*   name  ;

    const NP* lab_abs ;
    const NP* ppo_abs ;
    const NP* bis_abs ;

    const NP* lab_rem ;
    const NP* ppo_rem ;
    const NP* bis_rem ;

    const NP* lab_cmp ;
    const NP* lab_cmp_2 ;
    const NP* ppo_cmp ;
    const NP* bis_cmp ;

    const G4MaterialPropertyVector* lab_cmp_vec ;
    const G4MaterialPropertyVector* ppo_cmp_vec ;
    const G4MaterialPropertyVector* bis_cmp_vec ;

    const G4MaterialPropertyVector* lab_cmp_cdf ;
    const G4MaterialPropertyVector* ppo_cmp_cdf ;
    const G4MaterialPropertyVector* bis_cmp_cdf ;

    const NP* lab_cmp_icdf ;
    const NP* ppo_cmp_icdf ;
    const NP* bis_cmp_icdf ;

    const NP* icdf ;

    const NP* lab_wls ;
    const NP* ppo_wls ;
    const NP* bis_wls ;

    static U4ScintThree* Create(const NPFold* materials );
    U4ScintThree(const NPFold* fold, const char* name);
    std::string desc() const ;
    NPFold* make_fold() const ;
    void save(const char* base, const char* rel=nullptr ) const ;
};


inline U4ScintThree* U4ScintThree::Create(const NPFold* materials ) // static
{
    std::stringstream ss ;
    ss
       << ABSLENGTH_PROPS << ","
       << REEMISSIONPROB_PROPS << ","
       << COMPONENT_PROPS
       ;
    std::string PROPS = ss.str();


    std::vector<const NPFold*> subs ;
    std::vector<std::string> names ;
    materials->find_subfold_with_all_keys( subs, names, PROPS.c_str() );

    int num_subs = subs.size();
    int num_names = names.size() ;
    assert( num_subs == num_names );

    const char* name = num_names > 0 ? names[0].c_str() : nullptr ;
    const NPFold* sub = num_subs > 0 ? subs[0] : nullptr ;
    bool with_scint = name && sub ;

    return with_scint ? new U4ScintThree(sub, name) : nullptr ;
}


inline U4ScintThree::U4ScintThree(const NPFold* scint_, const char* name_)
    :
    num_wlsamp(ssys::getenvint("U4ScintThree__num_wlsamp", 1000)),
    scint(scint_),
    name(strdup(name_)),
    lab_abs(scint->get("ABSLENGTH")),
    ppo_abs(scint->get("PPOABSLENGTH")),
    bis_abs(scint->get("bisMSBABSLENGTH")),
    lab_rem(scint->get("REEMISSIONPROB")),
    ppo_rem(scint->get("PPOREEMISSIONPROB")),
    bis_rem(scint->get("bisMSBREEMISSIONPROB")),
    lab_cmp(scint->get("FASTCOMPONENT")),
    lab_cmp_2(scint->get("SLOWCOMPONENT")),
    ppo_cmp(scint->get("PPOCOMPONENT")),
    bis_cmp(scint->get("bisMSBCOMPONENT")),
    lab_cmp_vec(U4MaterialPropertyVector::FromArray(lab_cmp)),
    ppo_cmp_vec(U4MaterialPropertyVector::FromArray(ppo_cmp)),
    bis_cmp_vec(U4MaterialPropertyVector::FromArray(bis_cmp)),
    lab_cmp_cdf(U4ScintCommon::Integral(lab_cmp_vec)),
    ppo_cmp_cdf(U4ScintCommon::Integral(ppo_cmp_vec)),
    bis_cmp_cdf(U4ScintCommon::Integral(bis_cmp_vec)),
    lab_cmp_icdf(U4ScintCommon::CreateGeant4InterpolatedInverseCDF(lab_cmp_cdf,num_bins,hd_factor,name,energy_not_wavelength)),
    ppo_cmp_icdf(U4ScintCommon::CreateGeant4InterpolatedInverseCDF(ppo_cmp_cdf,num_bins,hd_factor,name,energy_not_wavelength)),
    bis_cmp_icdf(U4ScintCommon::CreateGeant4InterpolatedInverseCDF(bis_cmp_cdf,num_bins,hd_factor,name,energy_not_wavelength)),
    icdf(NP::Stack_(lab_cmp_icdf,ppo_cmp_icdf,bis_cmp_icdf)),
    lab_wls(U4ScintCommon::CreateWavelengthSamples(lab_cmp_cdf, num_wlsamp )),
    ppo_wls(U4ScintCommon::CreateWavelengthSamples(ppo_cmp_cdf, num_wlsamp )),
    bis_wls(U4ScintCommon::CreateWavelengthSamples(bis_cmp_cdf, num_wlsamp ))
{
}

inline std::string U4ScintThree::desc() const
{
    std::stringstream ss ;
    ss << "U4ScintThree::desc" << std::endl
       << " name " << name
       << " lab_abs " << ( lab_abs ? lab_abs->sstr() : "-" )
       << " ppo_abs " << ( ppo_abs ? ppo_abs->sstr() : "-" )
       << " bis_abs " << ( bis_abs ? bis_abs->sstr() : "-" )
       << " lab_rem " << ( lab_rem ? lab_rem->sstr() : "-" )
       << " ppo_rem " << ( ppo_rem ? ppo_rem->sstr() : "-" )
       << " bis_rem " << ( bis_rem ? bis_rem->sstr() : "-" )
       << " lab_cmp " << ( lab_cmp ? lab_cmp->sstr() : "-" )
       << " lab_cmp_2 " << ( lab_cmp_2 ? lab_cmp_2->sstr() : "-" )
       << " ppo_cmp " << ( ppo_cmp ? ppo_cmp->sstr() : "-" )
       << " bis_cmp " << ( bis_cmp ? bis_cmp->sstr() : "-" )
       << " lab_cmp_icdf " << ( lab_cmp_icdf ? lab_cmp_icdf->sstr() : "-" )
       << " ppo_cmp_icdf " << ( ppo_cmp_icdf ? ppo_cmp_icdf->sstr() : "-" )
       << " bis_cmp_icdf " << ( bis_cmp_icdf ? bis_cmp_icdf->sstr() : "-" )
       << " icdf "    << ( icdf ? icdf->sstr() : "-" )
       << " lab_wls " << ( lab_wls ? lab_wls->sstr() : "-" )
       << " ppo_wls " << ( ppo_wls ? ppo_wls->sstr() : "-" )
       << " bis_wls " << ( bis_wls ? bis_wls->sstr() : "-" )
       << std::endl
       ;

    std::string str = ss.str();
    return str ;
}

inline NPFold* U4ScintThree::make_fold() const
{
    NPFold* fold = new NPFold ;

    fold->add("lab_abs", lab_abs) ;
    fold->add("ppo_abs", ppo_abs) ;
    fold->add("bis_abs", bis_abs) ;

    fold->add("lab_rem", lab_rem) ;
    fold->add("ppo_rem", ppo_rem) ;
    fold->add("bis_rem", bis_rem) ;

    fold->add("lab_cmp", lab_cmp) ;
    fold->add("lab_cmp_2", lab_cmp_2) ;
    fold->add("ppo_cmp", ppo_cmp) ;
    fold->add("bis_cmp", bis_cmp) ;

    fold->add("lab_cmp_icdf", lab_cmp_icdf) ;
    fold->add("ppo_cmp_icdf", ppo_cmp_icdf) ;
    fold->add("bis_cmp_icdf", bis_cmp_icdf) ;

    fold->add("icdf", icdf) ;

    fold->add("lab_wls", lab_wls );
    fold->add("ppo_wls", ppo_wls );
    fold->add("bis_wls", bis_wls );

    return fold ;
}

inline void U4ScintThree::save(const char* base, const char* rel ) const
{
    NPFold* fold = make_fold();
    fold->save(base, rel);
}
