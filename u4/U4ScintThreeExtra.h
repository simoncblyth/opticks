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
    U4ScintThreeExtra(const NPFold* ls, const char* name);
    std::string desc() const ;
    NPFold* make_fold() const ;
    void save(const char* base, const char* rel=nullptr ) const ;

    const sdomain dom ;
    const NPFold* ls ;
    const char*   name  ;

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

    const NP* dcom ;         // value collection (com_abs_std, lab_frc_std, ppo_frc_std, bis_frc_std)
    const NP* fcom ;         // value collection narrowed to float




    const NP* lab_rem ;
    const NP* ppo_rem ;
    const NP* bis_rem ;

    const G4MaterialPropertyVector* lab_rem_vec ;
    const G4MaterialPropertyVector* ppo_rem_vec ;
    const G4MaterialPropertyVector* bis_rem_vec ;

    const NP* lab_rem_std ;
    const NP* ppo_rem_std ;
    const NP* bis_rem_std ;
    const NP* zer_rem_std ;
    const NP* drem ;
    const NP* frem ;

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

8. (com_abs_std,lab_frc_std,ppo_frc_std,bis_frc_std) => dcom

   * NP::MakePStack stack together values into double4 payload

9. dcom => fcom

   * NP::MakeNarrow for float4 payload

10. (lab_rem, ppo_rem, bis_rem) => (lab_rem_vec, ppo_rem_vec, bis_rem_vec)

   * U4MaterialPropertyVector::FromArray converted into Geant4 property vec
   * the vec use standard Geant4 MeV energy domain

11. (lab_rem_vec, ppo_rem_vec, bis_rem_vec) => (lab_rem_std, ppo_rem_std, bis_rem_std)

   * use Geant4 G4MaterialPropertyVector::Value to U4ScintThreeExtra::DomainInterpolate values into standard domain
   * despite domain energies needed for the Value args the standard arrays use wavelength_nm domain

12. (lab_rem_std, ppo_rem_std, bis_rem_std, zer_rem_std) => drem

   * NP::MakePStack stack together values into double4 payload

13. drem => frem

   * NP::MakeNarrow for float4 payload


**/


inline U4ScintThreeExtra::U4ScintThreeExtra(const NPFold* ls_, const char* name_)
    :
    ls(ls_),
    name(strdup(name_)),
    lab_abs(ls->get("ABSLENGTH")),
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
    dcom(NP::MakePStack(com_abs_std,lab_frc_std,ppo_frc_std,bis_frc_std)),  // one frac redundant as the frc add to 1.
    fcom(NP::MakeNarrow(dcom)),
    lab_rem(ls->get("REEMISSIONPROB")),
    ppo_rem(ls->get("PPOREEMISSIONPROB")),
    bis_rem(ls->get("bisMSBREEMISSIONPROB")),
    lab_rem_vec(U4MaterialPropertyVector::FromArray(lab_rem)),
    ppo_rem_vec(U4MaterialPropertyVector::FromArray(ppo_rem)),
    bis_rem_vec(U4MaterialPropertyVector::FromArray(bis_rem)),
    lab_rem_std(DomainInterpolate(lab_rem_vec, dom)),
    ppo_rem_std(DomainInterpolate(ppo_rem_vec, dom)),
    bis_rem_std(DomainInterpolate(bis_rem_vec, dom)),
    zer_rem_std(NP::MakePLikeWithValue(bis_rem_std, 0.)),
    drem(NP::MakePStack(lab_rem_std,ppo_rem_std,bis_rem_std,zer_rem_std)),
    frem(NP::MakeNarrow(drem))
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
       << " lab_rem     " << ( lab_rem     ? lab_rem->sstr()     : "-" ) << "\n"
       << " ppo_rem     " << ( ppo_rem     ? ppo_rem->sstr()     : "-" ) << "\n"
       << " bis_rem     " << ( bis_rem     ? bis_rem->sstr()     : "-" ) << "\n"
       << " lab_rem_std " << ( lab_rem_std ? lab_rem_std->sstr() : "-" ) << "\n"
       << " ppo_rem_std " << ( ppo_rem_std ? ppo_rem_std->sstr() : "-" ) << "\n"
       << " bis_rem_std " << ( bis_rem_std ? bis_rem_std->sstr() : "-" ) << "\n"
       << " zer_rem_std " << ( zer_rem_std ? zer_rem_std->sstr() : "-" ) << "\n"
       << " drem        " << ( drem        ? drem->sstr()        : "-" ) << "\n"
       << " frem        " << ( frem        ? frem->sstr()        : "-" ) << "\n"
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

    fold->add("lab_rem", lab_rem) ;
    fold->add("ppo_rem", ppo_rem) ;
    fold->add("bis_rem", bis_rem) ;

    fold->add("lab_rem_std", lab_rem_std) ;
    fold->add("ppo_rem_std", ppo_rem_std) ;
    fold->add("bis_rem_std", bis_rem_std) ;
    fold->add("zer_rem_std", zer_rem_std) ;
    fold->add("drem", drem ) ;
    fold->add("frem", frem ) ;

    return fold ;
}

inline void U4ScintThreeExtra::save(const char* base, const char* rel ) const
{
    NPFold* fold = make_fold();
    fold->save(base, rel);
}

