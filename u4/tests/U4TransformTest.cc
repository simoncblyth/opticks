
#include "G4PVPlacement.hh"

#include "U4RotationMatrix.h"
#include "U4Material.hh"
#include "U4VolumeMaker.hh"
#include "U4Transform.h"
#include "NP.hh"

#include <glm/gtx/string_cast.hpp>


G4VPhysicalVolume* MakePV()
{
    G4LogicalVolume* lv = U4VolumeMaker::Box_(1000., U4Material::VACUUM, "pfx" ); 

    G4ThreeVector tla(10., 20., 30.) ; 

    double phi = glm::pi<double>()/4. ; 
    U4RotationMatrix* rot = U4RotationMatrix::ZZ(phi); 

    const char* pv_name = "pv" ;
    G4LogicalVolume* mother_lv = nullptr ; 

    G4VPhysicalVolume* pv = new G4PVPlacement(rot, tla, lv, pv_name, mother_lv, false, 0);

    return pv ; 
}


int main(int argc, char** argv)
{
    const G4VPhysicalVolume* pv = MakePV(); 

    glm::tmat4x4<double> tr0(1.); 
    U4Transform::WriteObjectTransform( glm::value_ptr(tr0), pv) ;

    NP* a = NP::Make<double>(1,4,4); 

    U4Transform::WriteObjectTransform( a->values<double>(), pv) ;
    std::cout << glm::to_string(tr0) << std::endl ; 
    
    glm::tmat4x4<double> tr1(1.); 
    memcpy( glm::value_ptr(tr1), a->cvalues<double>() ,  16*sizeof(double) ); 
    std::cout << glm::to_string(tr1) << std::endl ; 

    return 0 ; 
}
