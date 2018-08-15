#pragma once

#include "X4_API_EXPORT.hh"

#include <string>
#include <glm/fwd.hpp>
#include "G4AffineTransform.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"

struct X4_API X4AffineTransform
{  
    static G4AffineTransform FromTransform(const G4Transform3D& T );
    static X4AffineTransform FromGLM( const glm::mat4& trs );

    X4AffineTransform( const G4AffineTransform&  t ); 

    G4RotationMatrix getRotation() const ;
    G4RotationMatrix getRotation_0() const ;
    std::string getRotationCode(const char* identifier) const ;

    G4ThreeVector getTranslation() const ;
    std::string getTranslationCode(const char* identifier) const ;

    const G4AffineTransform& tr ;    
    const G4RotationMatrix  rot ; 
    const G4ThreeVector     tla ;  

};
