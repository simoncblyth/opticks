
#include <sstream>
#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include "X4AffineTransform.hh"
#include "X4ThreeVector.hh"

X4AffineTransform::X4AffineTransform(const G4AffineTransform&  t ) 
    :
    tr(t),
    rot(tr.NetRotation()),
    tla(tr.NetTranslation())
{
}

bool X4AffineTransform::isIdentityRotation() const 
{
    return rot.isIdentity(); 
}
bool X4AffineTransform::isIdentityTranslation() const 
{
    return 
        tla.x() == 0. && 
        tla.y() == 0. && 
        tla.z() == 0. 
        ; 
}
bool X4AffineTransform::isIdentityTransform() const 
{
    return isIdentityRotation() && isIdentityTranslation() ;
}
 


G4AffineTransform X4AffineTransform::FromTransform(const G4Transform3D& T )
{
    // duplicate from CFG4.CMath

    G4ThreeVector colX(T.xx(), T.xy(), T.xz());
    G4ThreeVector colY(T.yx(), T.yy(), T.yz());
    G4ThreeVector colZ(T.zx(), T.zy(), T.zz());

    G4RotationMatrix rot(colX,colY,colZ) ;
    G4ThreeVector tlate(T.dx(), T.dy(), T.dz());

    return G4AffineTransform( rot, tlate) ; 
}



X4AffineTransform X4AffineTransform::FromGLM( const glm::mat4& trs )
{
    glm::mat3 trot(trs) ; 
    glm::mat3 trot_T = glm::transpose(trot) ; 
    glm::vec4 tlate = trs[3] ; 

    glm::vec3 _cx = trot_T[0] ;
    glm::vec3 _cy = trot_T[1] ;
    glm::vec3 _cz = trot_T[2] ;

    G4ThreeVector colX(_cx.x, _cx.y, _cx.z); 
    G4ThreeVector colY(_cy.x, _cy.y, _cy.z); 
    G4ThreeVector colZ(_cz.x, _cz.y, _cz.z); 

    G4RotationMatrix rot(colX, colY, colZ)  ; 

    G4ThreeVector tla(tlate.x,tlate.y,tlate.z);
 
    G4AffineTransform af( rot, tla ) ;  

    X4AffineTransform xaf(af); 

    return xaf ; 
}


G4RotationMatrix X4AffineTransform::getRotation_0() const 
{
    G4RotationMatrix rot2(rot.colX(), rot.colY(), rot.colZ())  ; 
    assert( rot2 == rot );  
    return rot2 ; 
}

G4RotationMatrix X4AffineTransform::getRotation() const 
{
    G4ThreeVector cx = rot.colX(); 
    G4ThreeVector cy = rot.colY(); 
    G4ThreeVector cz = rot.colZ(); 
 
    // testing a way of splaying out into the numbers for codegen 
    G4RotationMatrix rotation( 
            G4ThreeVector(cx.x(), cx.y(), cx.z()),
            G4ThreeVector(cy.x(), cy.y(), cy.z()),
            G4ThreeVector(cz.x(), cz.y(), cz.z())
            ) ; 
    assert( rotation == rot );  
    return rotation ; 
}

std::string X4AffineTransform::getRotationCode(const char* identifier) const 
{
    std::stringstream ss ; 
    ss << "G4RotationMatrix* " 
       << identifier 
       << " = new G4RotationMatrix"
       << "("
       << X4ThreeVector::Code(rot.colX(), NULL)
       << ","
       << X4ThreeVector::Code(rot.colY(), NULL)
       << ","
       << X4ThreeVector::Code(rot.colZ(), NULL)
       << ")"
       << ";"
    ;
    return ss.str(); 
}

G4ThreeVector X4AffineTransform::getTranslation() const 
{
    return G4ThreeVector( tla.x(), tla.y(), tla.z() ) ; 
}

std::string X4AffineTransform::getTranslationCode(const char* identifier) const 
{
    return X4ThreeVector::Code( tla, identifier ); 
}







