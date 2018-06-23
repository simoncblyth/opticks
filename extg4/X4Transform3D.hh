#pragma once

#include <string>
#include <array>
#include "X4_API_EXPORT.hh"
#include "G4Transform3D.hh"

class G4VPhysicalVolume ; 
class G4DisplacedSolid ; 

#include "glm/fwd.hpp"


/**
X4Transform3D
=================

Adopting:

* X4Transform3D::GetObjectTransform 
  (same as the M44T approach to mapping from G4Transform3D to glm) 
  succeeds to yield glTF render 
  that appears correct : PMTs pointing and assembled correctly  

* see issues/g4Live_gltf_shakedown.rst

* assembly of PMT within its "frame" (of 5 parts) only involves 
  translation in z : so getting that to appear correct can 
  be deceptive as no rotation   


NB majority of the methods only used in development
   to identify the correct mapping to use between 
   the transform descriptions of Geant4 and Opticks (GLM) 


**/

struct X4_API X4Transform3D
{
    static std::string Digest(const G4Transform3D&  transform ); 
    static G4Transform3D Convert( const glm::mat4&     trs );
    static glm::mat4     Convert(const G4Transform3D&  transform );

    static glm::mat4 GetObjectTransform(const G4VPhysicalVolume* const pv ); // <-- preferred
    static glm::mat4 GetFrameTransform(const G4VPhysicalVolume* const pv );
    static glm::mat4 GetDisplacementTransform(const G4DisplacedSolid* const disp );

    typedef enum { M43, M44, M44T } Mapping_t ; 
    typedef std::array<float,16> Array_t ; 
    static void copyToArray(Array_t& a, const G4Transform3D& t, Mapping_t mapping );
    static void copyToArray_M43(Array_t& a, const G4Transform3D& t );
    static void copyToArray_M44(Array_t& a, const G4Transform3D& t );
    static void copyToArray_M44T(Array_t& a, const G4Transform3D& t );
    static unsigned checkArray(const Array_t& a);

    const G4Transform3D&    t ; 
    Mapping_t               mapping ; 
    Array_t                 ar ; 
    glm::mat4*              mat ; 

    X4Transform3D(const G4Transform3D&  t, Mapping_t a = M44T);
    void init();
    void dump(const char* msg) const ;

    std::string digest() const ; 

}; 


