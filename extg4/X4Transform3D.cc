#include "G4VPhysicalVolume.hh"
#include "X4Transform3D.hh"
#include "SDigest.hh"

std::string X4Transform3D::Digest(const G4Transform3D&  transform)
{
    X4Transform3D xt(transform);
    return xt.digest() ; 
}

glm::mat4 X4Transform3D::Convert(const G4Transform3D&  transform)
{
    X4Transform3D xt(transform);
    return xt.tr ; 
}


glm::mat4* X4Transform3D::GetLocalTransform(const G4VPhysicalVolume* const pv)
{
    G4RotationMatrix rot = pv->GetObjectRotationValue() ;  // obj relative to mother
    G4ThreeVector    tla = pv->GetObjectTranslation() ;
    G4Transform3D    tra(rot,tla);
    glm::mat4* transform = new glm::mat4(Convert( tra ));
    return transform ; 
}





#ifdef X4_TRANSFORM_43

X4Transform3D::X4Transform3D(const G4Transform3D&  t) 
    :
    ar{{float(t.xx()),float(t.xy()),float(t.xz()),float(t.dx()),
        float(t.yx()),float(t.yy()),float(t.yz()),float(t.dy()),
        float(t.zx()),float(t.zy()),float(t.zz()),float(t.dz()),
        float(0),     float(0),     float(0),     float(1)}} ,
    tr(glm::make_mat4(ar.data()))
{
}

#else

X4Transform3D::X4Transform3D(const G4Transform3D&  t) 
    :
    ar{{float(t.xx()),float(t.xy()),float(t.xz()),float(0),
        float(t.yx()),float(t.yy()),float(t.yz()),float(0),
        float(t.zx()),float(t.zy()),float(t.zz()),float(0),
        float(t.dx()),float(t.dy()),float(t.dz()),float(1)}} ,
    tr(glm::make_mat4(ar.data()))
{
}

#endif



std::string X4Transform3D::digest() const 
{
    return SDigest::digest( (void*)this, sizeof(X4Transform3D) );
}



