#pragma once
/**
SVolume.h
=============

Developed within j/PMTSim but as generally useful moved to sysrap header-only.

An early user of this was x4/tests/X4IntersectVolumeTest.cc as used by x4/xxv.sh  

HUH: thats wierd, lots of passing around vectors of solids just to get their names 

**/

#include <vector>
#include <string>
#include <array>

struct NP ; 
class G4VSolid ; 
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 

struct SVolume
{
    static void SaveTransforms( std::vector<double>* tr, std::vector<G4VSolid*>* names, const char* path ); 
    static void SaveTransforms( std::vector<double>* tr, std::vector<G4VSolid*>* names, const char* fold, const char* name ); 

    static void GetSolidNames(std::vector<std::string>& names,  std::vector<G4VSolid*>* solids ); 

    static NP* MakeArray( std::vector<double>* tr, std::vector<G4VSolid*>* solids ); 
    static void DumpTransforms( std::vector<double>* tr, std::vector<G4VSolid*>* names, const char* msg ); 

    static void Traverse(const G4VPhysicalVolume* const pv, std::vector<double>* tr, std::vector<G4VSolid*>* names ); 
    static void Traverse_r(const G4VPhysicalVolume* const pv, int depth, std::vector<double>* tr, std::vector<G4VSolid*>* names); 
    static bool IsIdentityRotation(const std::array<double, 16>& a, double epsilon ) ; 
    static void GetObjectTransform(std::array<double, 16>& a, const G4VPhysicalVolume* const pv); 
    static void DumpSolids(); 
};


#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <array>

#include "NP.hh"

#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SolidStore.hh"
#include "G4DisplacedSolid.hh"
#include "G4Material.hh"


inline void SVolume::SaveTransforms( std::vector<double>* tr, std::vector<G4VSolid*>* solids, const char* fold, const char* name ) // static
{
    NP* a = MakeArray(tr, solids); 
    a->save(fold, name); 
}

inline void SVolume::SaveTransforms( std::vector<double>* tr, std::vector<G4VSolid*>* solids, const char* path ) // static
{
    NP* a = MakeArray(tr, solids); 
    a->save(path); 
}

inline void SVolume::GetSolidNames(std::vector<std::string>& names,  std::vector<G4VSolid*>* solids ) // static
{
    unsigned num = solids->size() ;
    for(unsigned i=0 ; i < num ; i++) 
    {
        G4VSolid* solid = (*solids)[i] ; 
        G4String name = solid->GetName() ; 
        names.push_back(name); 
    }
}

/**
SVolume::MakeArray
---------------------

Creates array of transforms with the names of the solids as metadata. 



**/
inline NP* SVolume::MakeArray( std::vector<double>* tr, std::vector<G4VSolid*>* solids ) // static
{
    unsigned num_solid = solids->size() ;
    unsigned num_value = tr->size(); 
    assert( num_value % 16 == 0 );  
    unsigned num_transform = num_value/num_solid/16 ; 

    bool expect = num_value == num_solid*num_transform*16 ; 

    if(!expect) std::cerr 
        << "SVolume::MakeArray"
        << " num_solid " << num_solid
        << " num_value " << num_value
        << " num_transform " << num_transform
        << " num_solid*num_transform*16 " << num_solid*num_transform*16
        << std::endl 
        ;
    assert(expect); 

    std::vector<std::string> names ; 
    GetSolidNames(names, solids) ; 

    NP* a = NP::Make<double>( num_solid, num_transform, 4, 4 );   // NB changed shape to allow multiple transforms per solid
    a->read( tr->data() ); 
    a->set_names(names) ;  // NB changed, to set_names formerly used a->meta string 
   
    return a ; 
}

inline void SVolume::DumpTransforms( std::vector<double>* tr, std::vector<G4VSolid*>* solids, const char* msg ) // static
{
    std::cout << msg << std::endl ; 
    unsigned num = solids->size() ;
    assert( tr->size() == num*16 ); 
    double* trd = tr->data() ; 

    for(unsigned i=0 ; i < num ; i++)
    {
        G4VSolid* solid = (*solids)[i] ; 
        G4String soname = solid->GetName() ; 
        std::cout 
            << std::setw(5) << i 
            << " : " 
            << std::setw(25) << soname 
            ;

        std::cout << " tr ( " ; 
        for(int j=0 ; j < 16 ; j++ ) 
            std::cout << std::fixed << std::setw(7) << std::setprecision(2) << trd[i*16+j] << " " ;  
        std::cout << ")" << std::endl ; 
    }
}

/**
SVolume::Traverse
-------------------

This is used for example by PMTFastSim::GetPV

**/

inline void SVolume::Traverse(const G4VPhysicalVolume* const pv, std::vector<double>* tr , std::vector<G4VSolid*>* solids ) // static
{
    Traverse_r( pv, 0, tr, solids); 
}

/**
SVolume::Traverse_r
---------------------

Recursively traverses the volume hierarchy collecting single level of transforms and solids. 
NB Assumes no multi-level transforms, so in this form its only appropriate for very simple volume trees.  

**/

inline void SVolume::Traverse_r(const G4VPhysicalVolume* const pv, int depth, std::vector<double>* tr, std::vector<G4VSolid*>* solids ) // static
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    G4VSolid* so = lv->GetSolid();
    const G4Material* const mt = lv->GetMaterial() ;
    G4String soname = so->GetName(); 

    std::array<double, 16> a ; 
    GetObjectTransform(a, pv); 
    if(tr) for(unsigned i=0 ; i < 16 ; i++) tr->push_back( a[i] ); 
    if(solids) solids->push_back(so); 

    std::cout 
        << "SVolume::Traverse_r"
        << " depth " << std::setw(2) << depth 
        << " pv " << std::setw(22) << pv->GetName()
        << " lv " << std::setw(22) << lv->GetName()
        << " so " << std::setw(22) << so->GetName()
        << " mt " << std::setw(15) << mt->GetName()
        ;

    bool identity_rot = IsIdentityRotation( a, 1e-6 ); 
    std::cout << " tr ( " ; 
    unsigned i0 = identity_rot ? 4*3 : 0 ; 
    for(unsigned i=i0 ; i < 16 ; i++) std::cout << std::fixed << std::setw(7) << std::setprecision(2) << a[i] << " " ; 
    std::cout << ") " << std::endl ; 

    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ ) 
    {
        const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
        Traverse_r( daughter_pv, depth+1, tr, solids );
    } 
}

inline bool SVolume::IsIdentityRotation(const std::array<double, 16>& a, double epsilon )
{
    unsigned not_identity=0 ; 
    for(unsigned i=0 ; i < 3 ; i++) 
    for(unsigned j=0 ; j < 3 ; j++)
    {
        unsigned idx = i*4 + j ; 
        double expect = i == j ? 1. : 0. ; 
        if( std::abs( a[idx] - expect) > epsilon ) not_identity += 1 ; 
    } 
    return not_identity == 0 ; 
}


inline void SVolume::GetObjectTransform(std::array<double, 16>& a, const G4VPhysicalVolume* const pv)
{
   // preferred for interop with glm/Opticks : obj relative to mother
    G4RotationMatrix rot = pv->GetObjectRotationValue() ;
    G4ThreeVector    tla = pv->GetObjectTranslation() ;
    G4Transform3D    t(rot,tla);

    a[ 0] = t.xx() ; a[ 1] = t.yx() ; a[ 2] = t.zx() ; a[ 3] = 0.f    ;
    a[ 4] = t.xy() ; a[ 5] = t.yy() ; a[ 6] = t.zy() ; a[ 7] = 0.f    ;
    a[ 8] = t.xz() ; a[ 9] = t.yz() ; a[10] = t.zz() ; a[11] = 0.f    ;
    a[12] = t.dx() ; a[13] = t.dy() ; a[14] = t.dz() ; a[15] = 1.f    ;
}

 
inline void SVolume::DumpSolids() // static
{
    G4SolidStore* store = G4SolidStore::GetInstance() ; 
    std::cout << "SVolume::DumpSolids G4SolidStore.size " << store->size() << std::endl ; 
    for( unsigned i=0 ; i < store->size() ; i++)
    {
        G4VSolid* solid = (*store)[i] ; 

        G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(solid) ; 
        G4VSolid* moved = disp ? disp->GetConstituentMovedSolid() : nullptr ; 

        std::cout 
            << " i " << std::setw(5) << i 
            << " solid.name " 
            << std::setw(30) << solid->GetName()
            << " moved.name " 
            << std::setw(30) << ( moved ? moved->GetName() : "-" )
            << std::endl
            ; 
    }
}

