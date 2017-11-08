

#include "COptical.hh"


G4OpticalSurfaceModel COptical::Model(unsigned model_)
{
   // materials/include/G4OpticalSurface.hh
    G4OpticalSurfaceModel model = unified ;
    switch(model_)
    {
       case 0:model = glisur   ; break; 
       case 1:model = unified  ; break; 
       case 2:model = LUT      ; break; 
       case 3:model = dichroic ; break; 
    }
    return model ; 
}   

G4OpticalSurfaceFinish COptical::Finish(unsigned finish_)
{
   // materials/include/G4OpticalSurface.hh
    G4OpticalSurfaceFinish finish = polished ;
    switch(finish_)
    {
        case 0:finish = polished              ;break;
        case 1:finish = polishedfrontpainted  ;break;  
        case 2:finish = polishedbackpainted   ;break;  
        case 3:finish = ground                ;break;
        case 4:finish = groundfrontpainted    ;break;  
        case 5:finish = groundbackpainted     ;break;  
    }
    return finish ; 
}


G4SurfaceType COptical::Type(unsigned type_)
{
    // materials/include/G4SurfaceProperty.hh
    G4SurfaceType type = dielectric_dielectric ;
    switch(type_)
    {
        case 0:type = dielectric_metal      ;break;
        case 1:type = dielectric_dielectric ;break;
    }
    return type ; 
}



