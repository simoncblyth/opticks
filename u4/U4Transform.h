#pragma once

#include <array>
#include "G4VPhysicalVolume.hh"
#include "G4Transform3D.hh"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

struct U4Transform
{
    static void WriteTransform(      double* dst, const glm::tmat4x4<double>& src ); 
    static void WriteObjectTransform(double* dst, const G4VPhysicalVolume* const pv) ;  // MOST USED BY OPTICKS
    static void WriteFrameTransform( double* dst, const G4VPhysicalVolume* const pv) ; 

    static void GetObjectTransform(glm::tmat4x4<double>& tr, const G4VPhysicalVolume* const pv) ; // MOST USED BY OPTICKS
    static void GetFrameTransform( glm::tmat4x4<double>& tr, const G4VPhysicalVolume* const pv) ; 

    template<typename T>
    static void Convert(glm::tmat4x4<T>& dst,  const G4Transform3D& src ); 

    template<typename T>
    static unsigned Check(const std::array<T,16>& a); 
};

void U4Transform::WriteTransform( double* dst, const glm::tmat4x4<double>& src )
{
    memcpy( dst, glm::value_ptr(src), sizeof(double)*16 );  
}
void U4Transform::WriteObjectTransform(double* dst, const G4VPhysicalVolume* const pv)
{
    glm::tmat4x4<double> tr(1.); 
    GetObjectTransform(tr, pv ); 
    WriteTransform(dst, tr ); 
}
void U4Transform::WriteFrameTransform(double* dst, const G4VPhysicalVolume* const pv)
{
    glm::tmat4x4<double> tr(1.); 
    GetFrameTransform(tr, pv ); 
    WriteTransform(dst, tr ); 
}
void U4Transform::GetObjectTransform(glm::tmat4x4<double>& tr, const G4VPhysicalVolume* const pv) 
{
   // preferred for interop with glm/Opticks : obj relative to mother
    G4RotationMatrix rot = pv->GetObjectRotationValue() ; 
    G4ThreeVector    tla = pv->GetObjectTranslation() ;
    G4Transform3D    tra(rot,tla);
    Convert(tr, tra);
}
void U4Transform::GetFrameTransform(glm::tmat4x4<double>& tr, const G4VPhysicalVolume* const pv) 
{
    const G4RotationMatrix* rotp = pv->GetFrameRotation() ;
    G4ThreeVector    tla = pv->GetFrameTranslation() ;
    G4Transform3D    tra(rotp ? *rotp : G4RotationMatrix(),tla);
    Convert(tr, tra ); 
}
template<typename T>
void U4Transform::Convert(glm::tmat4x4<T>& d,  const G4Transform3D& s ) // static
{
    // M44T
    T zero(0.); 
    T one(1.); 
    std::array<T, 16> a = {{
             s.xx(), s.yx(), s.zx(), zero ,  
             s.xy(), s.yy(), s.zy(), zero ,
             s.xz(), s.yz(), s.zz(), zero ,    
             s.dx(), s.dy(), s.dz(), one   }} ; 

    unsigned n = Check(a);
    assert( n == 0);
    memcpy( glm::value_ptr(d), a.data(), sizeof(T)*16 );  
}

template<typename T>
unsigned U4Transform::Check(const std::array<T,16>& a) // static
{
    unsigned num_nan(0);
    unsigned num_inf(0);
    for(unsigned i=0 ; i < 16 ; i++) if(std::isnan(a[i])) num_nan++ ;
    for(unsigned i=0 ; i < 16 ; i++) if(std::isinf(a[i])) num_inf++ ;
    return num_nan + num_inf ;
}

