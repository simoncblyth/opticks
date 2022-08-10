
#include <vector>
#include "G4PVPlacement.hh"

#include "U4RotationMatrix.h"
#include "U4Material.hh"
#include "U4VolumeMaker.hh"
#include "U4Transform.h"
#include "NP.hh"
#include "strid.h"

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

void test_Write()
{
    const G4VPhysicalVolume* pv = MakePV(); 

    {
        glm::tmat4x4<double> tr0(1.); 
        glm::tmat4x4<double> tr1(1.); 
        U4Transform::WriteObjectTransform( glm::value_ptr(tr0), pv) ;
        U4Transform::WriteFrameTransform(  glm::value_ptr(tr1), pv) ;
        std::cout << glm::to_string(tr0) << std::endl ; 
        std::cout << glm::to_string(tr1) << std::endl ; 
    }

    {
        NP* a = NP::Make<double>(2,4,4); 
        U4Transform::WriteObjectTransform( a->values<double>() + 0  , pv) ;
        U4Transform::WriteFrameTransform(  a->values<double>() + 16 , pv) ;
        
        std::vector<glm::tmat4x4<double>> m(2) ; 
        memcpy((double*)m.data(), a->cvalues<double>() ,  a->arr_bytes() ); 

        std::cout << glm::to_string(m[0]) << std::endl ; 
        std::cout << glm::to_string(m[1]) << std::endl ; 
    }
}


void test_Get()
{
    const G4VPhysicalVolume* pv = MakePV(); 

    glm::tmat4x4<double> m2w(1.) ;  
    U4Transform::GetObjectTransform(m2w, pv); 

    glm::tmat4x4<double> w2m(1.) ;  
    U4Transform::GetFrameTransform(w2m, pv); 

    glm::tmat4x4<double> m2w_w2m = m2w * w2m ;  

    std::cout << strid::Desc_("m2w", "w2m", "m2w_w2m", m2w, w2m, m2w_w2m ) << std::endl ; 
}



int main(int argc, char** argv)
{
    //test_Get(); 
    test_Write(); 

    return 0 ; 
}
