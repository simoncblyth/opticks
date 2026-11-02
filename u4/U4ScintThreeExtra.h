#pragma once
/**
U4ScintThreeExtra.h
====================

See::

    ~/o/u4/tests/U4ScintThreeExtra_test.sh


**/

#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "NPFold.h"
#include "ssys.h"
#include "sdomain.h"

#include "U4MaterialPropertyVector.h"

struct U4ScintThreeExtra
{
    static NP* DomainInterpolate(const G4MaterialPropertyVector* vec, const sdomain& dom);
    static NP* MakePropArray(const NP* dcom, const NP* drem, int ls_material_index=-1);
    U4ScintThreeExtra(const NPFold* ls, const char* name, int ls_material_index=-1);
    std::string desc() const ;
    NPFold* make_fold() const ;
    void save(const char* base, const char* rel=nullptr ) const ;

    const sdomain dom ;
    const NPFold* ls ;
    const char*   name  ;
    const int     ls_material_index ;

    const NP* lab_abs ;
    const NP* ppo_abs ;
    const NP* bis_abs ;

    const G4MaterialPropertyVector* lab_abs_vec ;
    const G4MaterialPropertyVector* ppo_abs_vec ;
    const G4MaterialPropertyVector* bis_abs_vec ;

    const NP* lab_abs_std ;
    const NP* ppo_abs_std ;
    const NP* bis_abs_std ;

    const NP* lab_coe_std ;  //          1 / lab_abs_std
    const NP* ppo_coe_std ;  //          1 / ppo_abs_std
    const NP* bis_coe_std ;  //          1 / bis_abs_std
    const NP* com_coe_std ;  // lab_coe_std + ppo_coe_std + bis_coe_std

    const NP* com_abs_std ;  //           1 / com_coe_std
    const NP* lab_frc_std ;  // lab_coe_std / com_coe_std
    const NP* ppo_frc_std ;  // ppo_coe_std / com_coe_std
    const NP* bis_frc_std ;  // bis_coe_std / com_coe_std

    const NP* com_frc_std ;  // lab_frc_std + ppo_frc_std + bis_frc_std   ( values should be 1. )

    const NP* dcom ;         // (nk,4) value collection (lab_frc_std, ppo_frc_std, bis_frc_std, com_abs_std)
    const NP* fcom ;         // value collection narrowed to float

    const NP* ppo_rem ;
    const NP* bis_rem ;

    const G4MaterialPropertyVector* ppo_rem_vec ;
    const G4MaterialPropertyVector* bis_rem_vec ;

    const NP* ppo_rem_std ;
    const NP* bis_rem_std ;
    const NP* zer_rem_std ;
    const NP* drem ;
    const NP* frem ;

    const NP* prop_three ;   // (2,nk,4) GPU compact payload : row0 (lab_frc,ppo_frc,bis_frc,com_abs), row1 (ppo_rem,bis_rem,0,0)

};


/**
U4ScintThreeExtra::DomainInterpolate
------------------------------------

Note that the property vec Value method requires an energy argument but the
domain of the property is the corresponding wavelength in nm as that
is what is needed for the domain of the GPU textures into which
these arrays are destined for.

**/

inline NP* U4ScintThreeExtra::DomainInterpolate(const G4MaterialPropertyVector* vec,  const sdomain& dom) // static
{
    NP* a = NP::Make<double>(dom.length, 2);
    double* aa = a->values<double>();
    for(int i=0 ; i < dom.length ; i++)
    {
        aa[i*2+0] = dom.wavelength_nm[i] ;
        aa[i*2+1] = vec->Value(dom.energy_eV[i]*eV)  ;
    }
    return a ;
}

/**
U4ScintThreeExtra::MakePropArray
---------------------------------

Gather the two (nk,4) collections into the (2,nk,4) prop_three array consumed
by the GPU side, with the domain and LS material index metas::

    row0 = ( fraction_lab, fraction_ppo, fraction_bis, abslength_combined )
    row1 = ( reemission_prob_ppo, reemission_prob_bis, 0., 0. )

**/

inline NP* U4ScintThreeExtra::MakePropArray(const NP* dcom, const NP* drem, int ls_material_index) // static
{
    if(dcom == nullptr || drem == nullptr) return nullptr ;

    const int nk = dcom->shape[0] ;
    if(drem->shape[0] != nk || dcom->shape[1] != 4 || drem->shape[1] != 4) return nullptr ;

    NP* prop_three = NP::Make<double>(2, nk, 4) ;
    double* vv = prop_three->values<double>() ;

    const double* com = dcom->cvalues<double>() ;
    const double* rem = drem->cvalues<double>() ;
    for(int i=0 ; i < nk*4 ; i++)
    {
        vv[i]        = com[i] ;
        vv[nk*4 + i] = rem[i] ;
    }

    prop_three->set_meta<float>("domain_low",   sdomain::DomainLow());
    prop_three->set_meta<float>("domain_high",  sdomain::DomainHigh());
    prop_three->set_meta<float>("domain_step",  sdomain::DomainStep());
    prop_three->set_meta<float>("domain_range", sdomain::DomainRange());

    if(ls_material_index >= 0) prop_three->set_meta<int>("ls_material_index", ls_material_index);

    return prop_three ;
}

/**
U4ScintThreeExtra::U4ScintThreeExtra
-------------------------------------

1. (lab_abs, ppo_abs, bis_abs) => (lab_abs_vec, ppo_abs_vec, bis_abs_vec)

   * U4MaterialPropertyVector::FromArray converted into Geant4 property vec
   * the vec use standard Geant4 MeV energy domain

2. (lab_abs_vec, ppo_abs_vec, bis_abs_vec) => (lab_abs_std, ppo_abs_std, bis_abs_std)

   * use Geant4 G4MaterialPropertyVector::Value to U4ScintThreeExtra::DomainInterpolate values into standard domain
   * despite domain energies needed for the Value args the standard arrays use wavelength_nm domain

3. (lab_abs_std, ppo_abs_std, bis_abs_std) =>  (lab_coe_std, ppo_coe_std, bis_coe_std)

   * absorption lengths across the standard domain are NP::MakePInverse inverted to give absorption coefficients

4. (lab_coe_std, ppo_coe_std, bis_coe_std) =>  com_coe_std

   * sum the absorption coefficient to give combined coefficient using NP::MakePSum

5. com_coe_std => com_abs_std

   * NP::MakePInverse invert the combined absorption coefficient to give a combined absorption length

6. (lab_coe_std,  ppo_coe_std, bis_coe_std,  com_coe_std) =>  (lab_frc_std, ppo_frc_std, bis_frc_std)

   * NP::MakePRatio form ratios of the three absorption coefficients with the combined sum

7.  (lab_frc_std, ppo_frc_std, bis_frc_std) =>  com_frc_std

   * NP::MakePSum sum the three fractions as check that values are close to 1.

8. (lab_frc_std, ppo_frc_std, bis_frc_std, com_abs_std) => dcom

   * NP::MakePStack stack together values into double4 payload

9. dcom => fcom

   * NP::MakeNarrow for float4 payload

10. (ppo_rem, bis_rem) => (ppo_rem_vec, bis_rem_vec)

   * U4MaterialPropertyVector::FromArray converted into Geant4 property vec
   * the vec use standard Geant4 MeV energy domain

11. (ppo_rem_vec, bis_rem_vec) => (ppo_rem_std, bis_rem_std)

   * use Geant4 G4MaterialPropertyVector::Value to U4ScintThreeExtra::DomainInterpolate values into standard domain
   * despite domain energies needed for the Value args the standard arrays use wavelength_nm domain

12. (ppo_rem_std, bis_rem_std, zer_rem_std, zer_rem_std) => drem

   * NP::MakePStack stack together values into double4 payload

13. drem => frem

   * NP::MakeNarrow for float4 payload

14. (dcom, drem) => prop_three (2,nk,4) with domain and ls_material_index metas

   * the compact array consumed by the GPU side textures


**/


inline U4ScintThreeExtra::U4ScintThreeExtra(const NPFold* ls_, const char* name_, int ls_material_index_)
    :
    ls(ls_),
    name(strdup(name_)),
    ls_material_index(ls_material_index_),
    lab_abs(ls->get("LABABSLENGTH")),
    ppo_abs(ls->get("PPOABSLENGTH")),
    bis_abs(ls->get("bisMSBABSLENGTH")),
    lab_abs_vec(U4MaterialPropertyVector::FromArray(lab_abs)),
    ppo_abs_vec(U4MaterialPropertyVector::FromArray(ppo_abs)),
    bis_abs_vec(U4MaterialPropertyVector::FromArray(bis_abs)),
    lab_abs_std(DomainInterpolate(lab_abs_vec, dom)),
    ppo_abs_std(DomainInterpolate(ppo_abs_vec, dom)),
    bis_abs_std(DomainInterpolate(bis_abs_vec, dom)),
    lab_coe_std(NP::MakePInverse(lab_abs_std)),
    ppo_coe_std(NP::MakePInverse(ppo_abs_std)),
    bis_coe_std(NP::MakePInverse(bis_abs_std)),
    com_coe_std(NP::MakePSum( lab_coe_std, ppo_coe_std, bis_coe_std )),
    com_abs_std(NP::MakePInverse(com_coe_std)),
    lab_frc_std(NP::MakePRatio( lab_coe_std, com_coe_std )),
    ppo_frc_std(NP::MakePRatio( ppo_coe_std, com_coe_std )),
    bis_frc_std(NP::MakePRatio( bis_coe_std, com_coe_std )),
    com_frc_std(NP::MakePSum( lab_frc_std, ppo_frc_std, bis_frc_std )),    // values should be close to 1. 
    dcom(NP::MakePStack(lab_frc_std,ppo_frc_std,bis_frc_std,com_abs_std)),  // one frac redundant as the frc add to 1.
    fcom(NP::MakeNarrow(dcom)),
    ppo_rem(ls->get("PPOREEMISSIONPROB")),
    bis_rem(ls->get("bisMSBREEMISSIONPROB")),
    ppo_rem_vec(U4MaterialPropertyVector::FromArray(ppo_rem)),
    bis_rem_vec(U4MaterialPropertyVector::FromArray(bis_rem)),
    ppo_rem_std(DomainInterpolate(ppo_rem_vec, dom)),
    bis_rem_std(DomainInterpolate(bis_rem_vec, dom)),
    zer_rem_std(NP::MakePLikeWithValue(bis_rem_std, 0.)),
    drem(NP::MakePStack(ppo_rem_std,bis_rem_std,zer_rem_std,zer_rem_std)),
    frem(NP::MakeNarrow(drem)),
    prop_three(MakePropArray(dcom, drem, ls_material_index))
{
}


inline std::string U4ScintThreeExtra::desc() const
{
    std::stringstream ss ;
    ss 
       << "[U4ScintThreeExtra::desc" << "\n"
       << " name        " << name << "\n"
       << " lab_abs     " << ( lab_abs     ? lab_abs->sstr()     : "-" ) << "\n"
       << " ppo_abs     " << ( ppo_abs     ? ppo_abs->sstr()     : "-" ) << "\n"
       << " bis_abs     " << ( bis_abs     ? bis_abs->sstr()     : "-" ) << "\n"
       << " lab_abs_std " << ( lab_abs_std ? lab_abs_std->sstr() : "-" ) << "\n"
       << " ppo_abs_std " << ( ppo_abs_std ? ppo_abs_std->sstr() : "-" ) << "\n"
       << " bis_abs_std " << ( bis_abs_std ? bis_abs_std->sstr() : "-" ) << "\n"
       << " lab_coe_std " << ( lab_coe_std ? lab_coe_std->sstr() : "-" ) << "\n"
       << " ppo_coe_std " << ( ppo_coe_std ? ppo_coe_std->sstr() : "-" ) << "\n"
       << " bis_coe_std " << ( bis_coe_std ? bis_coe_std->sstr() : "-" ) << "\n"
       << " com_coe_std " << ( com_coe_std ? com_coe_std->sstr() : "-" ) << "\n"
       << " com_abs_std " << ( com_abs_std ? com_abs_std->sstr() : "-" ) << "\n"
       << " lab_frc_std " << ( lab_frc_std ? lab_frc_std->sstr() : "-" ) << "\n"
       << " ppo_frc_std " << ( ppo_frc_std ? ppo_frc_std->sstr() : "-" ) << "\n"
       << " bis_frc_std " << ( bis_frc_std ? bis_frc_std->sstr() : "-" ) << "\n"
       << " com_frc_std " << ( com_frc_std ? com_frc_std->sstr() : "-" ) << "\n"
       << " dcom        " << ( dcom        ? dcom->sstr()        : "-" ) << "\n"
       << " fcom        " << ( fcom        ? fcom->sstr()        : "-" ) << "\n"
       << " ppo_rem     " << ( ppo_rem     ? ppo_rem->sstr()     : "-" ) << "\n"
       << " bis_rem     " << ( bis_rem     ? bis_rem->sstr()     : "-" ) << "\n"
       << " ppo_rem_std " << ( ppo_rem_std ? ppo_rem_std->sstr() : "-" ) << "\n"
       << " bis_rem_std " << ( bis_rem_std ? bis_rem_std->sstr() : "-" ) << "\n"
       << " zer_rem_std " << ( zer_rem_std ? zer_rem_std->sstr() : "-" ) << "\n"
       << " drem        " << ( drem        ? drem->sstr()        : "-" ) << "\n"
       << " frem        " << ( frem        ? frem->sstr()        : "-" ) << "\n"
       << " prop_three  " << ( prop_three  ? prop_three->sstr()  : "-" ) << "\n"
       << "]U4ScintThreeExtra::desc" << "\n"
       ;
    std::string str = ss.str();
    return str ;
}

inline NPFold* U4ScintThreeExtra::make_fold() const
{
    NPFold* fold = new NPFold ;

    fold->add("lab_abs", lab_abs) ;
    fold->add("ppo_abs", ppo_abs) ;
    fold->add("bis_abs", bis_abs) ;

    fold->add("lab_abs_std", lab_abs_std) ;
    fold->add("ppo_abs_std", ppo_abs_std) ;
    fold->add("bis_abs_std", bis_abs_std) ;

    fold->add("lab_coe_std", lab_coe_std) ;
    fold->add("ppo_coe_std", ppo_coe_std) ;
    fold->add("bis_coe_std", bis_coe_std) ;
    fold->add("com_coe_std", com_coe_std) ;

    fold->add("com_abs_std", com_abs_std) ;
    fold->add("lab_frc_std", lab_frc_std) ;
    fold->add("ppo_frc_std", ppo_frc_std) ;
    fold->add("bis_frc_std", bis_frc_std) ;
    fold->add("com_frc_std", com_frc_std) ;

    fold->add("dcom", dcom ) ;
    fold->add("fcom", fcom ) ;

    fold->add("ppo_rem", ppo_rem) ;
    fold->add("bis_rem", bis_rem) ;

    fold->add("ppo_rem_std", ppo_rem_std) ;
    fold->add("bis_rem_std", bis_rem_std) ;
    fold->add("zer_rem_std", zer_rem_std) ;
    fold->add("drem", drem ) ;
    fold->add("frem", frem ) ;

    fold->add("prop_three", prop_three ) ;

    return fold ;
}

inline void U4ScintThreeExtra::save(const char* base, const char* rel ) const
{
    NPFold* fold = make_fold();
    fold->save(base, rel);
}

