#pragma once
/**
U4ScintThree.h : three species LAB/PPO/bisMSB LS model
========================================================

CPU-side data preparation for the three-species LS model.
Only PPO and bisMSB emit photons: LAB is the solvent (absorbs + non-radiative
transfer to PPO/bisMSB). Reads from the LS material subfold::

    LABABSLENGTH.npy
    PPOABSLENGTH.npy     PPOCOMPONENT.npy     PPOREEMISSIONPROB.npy
    bisMSBABSLENGTH.npy  bisMSBCOMPONENT.npy  bisMSBREEMISSIONPROB.npy

The COMPONENT curves feed MakeICDFArray into icdf_three and the
MaterialConstProperty values are scanned by MakeConstArray into const_three.
The abslength and reemission curves are handled by U4ScintThreeExtra which
forms the compact prop_three array.

**/

#include <iostream>
#include <string>
#include <iomanip>

#include "Randomize.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "ssys.h"
#include "sdomain.h"
#include "NPFold.h"
#include "NPX.h"
#include "U4MaterialPropertyVector.h"
#include "U4ScintCommon.h"

struct U4ScintThree
{
    static constexpr const bool VERBOSE = false ;
    static constexpr const char* ABSLENGTH_PROPS = "LABABSLENGTH,PPOABSLENGTH,bisMSBABSLENGTH" ;
    static constexpr const char* REEMISSIONPROB_PROPS = "PPOREEMISSIONPROB,bisMSBREEMISSIONPROB" ;
    static constexpr const char* COMPONENT_PROPS = "PPOCOMPONENT,bisMSBCOMPONENT" ;


    static const int num_bins = 4096 ;
    static const int hd_factor = 20 ;
    static const bool energy_not_wavelength = false ;



    int   num_wlsamp ;
    const NPFold* scint ;
    const char*   name  ;

    const NP* lab_abs ;
    const NP* ppo_abs ;
    const NP* bis_abs ;

    const NP* ppo_rem ;
    const NP* bis_rem ;

    const NP* padded_ppo_cmp ;
    const NP* padded_bis_cmp ;
    const NP* ppo_cmp ;
    const NP* bis_cmp ;

    const G4MaterialPropertyVector* ppo_cmp_vec ;
    const G4MaterialPropertyVector* bis_cmp_vec ;

    const G4MaterialPropertyVector* ppo_cmp_cdf ;
    const G4MaterialPropertyVector* bis_cmp_cdf ;

    const NP* ppo_cmp_icdf ;
    const NP* bis_cmp_icdf ;

    const NP* icdf_three ;
    const NP* const_three ;    // six LS new-optical-model constants scanned from the LS MaterialConstProperty

    const long seed ;
    const NP* ppo_wls ;
    const NP* bis_wls ;
    const NP* wls ;

    static U4ScintThree* Create(const NPFold* materials );
    static NP* MakeICDFArray( const NP* ppo_cmp_icdf, const NP* bis_cmp_icdf );
    static NP* MakeConstArray( const NPFold* scint_ );
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


/**
U4ScintThree::U4ScintThree
---------------------------

Do everything CTOR that operates in triplicate:

1. access scint material props from argument NPFold
2. remove property zero padding with NP::MakePCopyStripZeroPadding
3. convert arrays into Geant4 material property vector with U4MaterialPropertyVector::FromArray
4. integrate to form CDF with U4ScintCommon::Integral
5. invert CDF with U4ScintCommon::CreateGeant4InterpolatedInverseCDF
6. create wavelength samples with U4ScintCommon::CreateWavelengthSamples

   * U4ScintCommon::CreateWavelengthSamples is the standard Geant4 way to generate wavelength sample from the CDF
   * wls samples stacked below are compared with Opticks GPU samples in ~/o/qudarap/tests/QScintThree_test.sh

TODO: clarify dependency ordering (wls does not depend on icdf)


**/

inline U4ScintThree::U4ScintThree(const NPFold* scint_, const char* name_)
    :
    num_wlsamp(ssys::getenvsizespec("U4ScintThree__num_wlsamp", "M1")),
    scint(scint_),
    name(strdup(name_)),
    lab_abs(scint->get("LABABSLENGTH")),
    ppo_abs(scint->get("PPOABSLENGTH")),
    bis_abs(scint->get("bisMSBABSLENGTH")),
    ppo_rem(scint->get("PPOREEMISSIONPROB")),
    bis_rem(scint->get("bisMSBREEMISSIONPROB")),
    padded_ppo_cmp(scint->get("PPOCOMPONENT")),
    padded_bis_cmp(scint->get("bisMSBCOMPONENT")),
    ppo_cmp(NP::MakePCopyStripZeroPadding(padded_ppo_cmp)),
    bis_cmp(NP::MakePCopyStripZeroPadding(padded_bis_cmp)),
    ppo_cmp_vec(U4MaterialPropertyVector::FromArray(ppo_cmp)),
    bis_cmp_vec(U4MaterialPropertyVector::FromArray(bis_cmp)),
    ppo_cmp_cdf(U4ScintCommon::Integral(ppo_cmp_vec)),
    bis_cmp_cdf(U4ScintCommon::Integral(bis_cmp_vec)),
    ppo_cmp_icdf(U4ScintCommon::CreateGeant4InterpolatedInverseCDF(ppo_cmp_cdf,num_bins,hd_factor,name,energy_not_wavelength)),
    bis_cmp_icdf(U4ScintCommon::CreateGeant4InterpolatedInverseCDF(bis_cmp_cdf,num_bins,hd_factor,name,energy_not_wavelength)),
    icdf_three(MakeICDFArray(ppo_cmp_icdf, bis_cmp_icdf)),  // (2,3,4096,1): emitting species rows 0:PPO 1:bisMSB
    const_three(MakeConstArray(scint_)),            // six LS new-optical-model constants for the qscint_three device struct
    seed(42),
    ppo_wls(U4ScintCommon::CreateWavelengthSamples<float>(ppo_cmp_cdf, num_wlsamp, seed )),
    bis_wls(U4ScintCommon::CreateWavelengthSamples<float>(bis_cmp_cdf, num_wlsamp, seed )),
    wls(NP::Stack_(ppo_wls,bis_wls))                // (2,num_wlsamp)
{
}

/**
U4ScintThree::MakeICDFArray
----------------------------

Stack the two per-emitting-species (3,4096,1) inverse CDF arrays into the
(2,3,4096,1) layered atlas of the 2 emitting species (rows 0:PPO 1:bisMSB)
consumed by the QScintThree layered texture.

**/

inline NP* U4ScintThree::MakeICDFArray( const NP* ppo_cmp_icdf, const NP* bis_cmp_icdf ) // static
{
    return NP::Stack_(ppo_cmp_icdf, bis_cmp_icdf) ;
}

/**
U4ScintThree::MakeConstArray
-----------------------------

Read the six LS new-optical-model constants from the material const properties
(the "MaterialConstProperty" array of the LS subfold, names in its names list)
and return them as a (N,) double array with names.
U4Tree::initScintThree stashes it into SSim/extra/jnm (const_three.npy)
and it is read on the GPU side by QScintThree::MakeDevInstance into the qscint_three struct.

**/

inline NP* U4ScintThree::MakeConstArray( const NPFold* scint_ ) // static
{
    const NP* cpa = scint_->get("MaterialConstProperty") ;
    if(cpa == nullptr)
    {
        std::cerr << "U4ScintThree::MakeConstArray no MaterialConstProperty in LS subfold\n" ;
        return nullptr ;
    }

    const char* keys[6] = {
        "LAB2PPO_PROB", "LAB2BIS_PROB",
        "LAB2PPO_TAU",  "LAB2BIS_TAU",
        "PPO_INTRINSICTAU", "BISMSB_INTRINSICTAU" } ;

    NPX::KV<double> kv ;
    for(int i=0 ; i < 6 ; i++)
    {
        const char* key = keys[i] ;
        NP::INT idx = cpa->get_name_index(key) ;
        if(idx < 0)
        {
            std::cerr << "U4ScintThree::MakeConstArray missing const property [" << key << "]\n" ;
            continue ;
        }
        double value = cpa->cvalues<double>()[idx] ;
        kv.add(key, value );
    }

    return kv.values() ;    // (N,) double with names, nullptr when none found
}

inline std::string U4ScintThree::desc() const
{
    std::stringstream ss ;
    ss << "U4ScintThree::desc" << std::endl
       << " name " << name
       << " lab_abs " << ( lab_abs ? lab_abs->sstr() : "-" )
       << " ppo_abs " << ( ppo_abs ? ppo_abs->sstr() : "-" )
       << " bis_abs " << ( bis_abs ? bis_abs->sstr() : "-" )
       << " ppo_rem " << ( ppo_rem ? ppo_rem->sstr() : "-" )
       << " bis_rem " << ( bis_rem ? bis_rem->sstr() : "-" )
       << " padded_ppo_cmp " << ( padded_ppo_cmp ? padded_ppo_cmp->sstr() : "-" )
       << " padded_bis_cmp " << ( padded_bis_cmp ? padded_bis_cmp->sstr() : "-" )
       << " ppo_cmp " << ( ppo_cmp ? ppo_cmp->sstr() : "-" )
       << " bis_cmp " << ( bis_cmp ? bis_cmp->sstr() : "-" )
       << " ppo_cmp_icdf " << ( ppo_cmp_icdf ? ppo_cmp_icdf->sstr() : "-" )
       << " bis_cmp_icdf " << ( bis_cmp_icdf ? bis_cmp_icdf->sstr() : "-" )
       << " icdf_three " << ( icdf_three ? icdf_three->sstr() : "-" )
       << " const_three " << ( const_three ? const_three->sstr() : "-" )
       << " seed " << seed
       << " ppo_wls " << ( ppo_wls ? ppo_wls->sstr() : "-" )
       << " bis_wls " << ( bis_wls ? bis_wls->sstr() : "-" )
       << " wls "   << ( wls ? wls->sstr() : "-" )
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

    fold->add("ppo_rem", ppo_rem) ;
    fold->add("bis_rem", bis_rem) ;

    fold->add("padded_ppo_cmp", padded_ppo_cmp) ;
    fold->add("padded_bis_cmp", padded_bis_cmp) ;

    fold->add("ppo_cmp", ppo_cmp) ;
    fold->add("bis_cmp", bis_cmp) ;

    fold->add("ppo_cmp_icdf", ppo_cmp_icdf) ;
    fold->add("bis_cmp_icdf", bis_cmp_icdf) ;

    fold->add("icdf_three", icdf_three) ;
    fold->add("const_three", const_three) ;
    fold->add("wls", wls );

    return fold ;
}

inline void U4ScintThree::save(const char* base, const char* rel ) const
{
    NPFold* fold = make_fold();
    fold->save(base, rel);
}
