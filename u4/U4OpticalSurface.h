#pragma once
/**
U4OpticalSurface.h
=====================

NB G4OpticalSurface ISA G4SurfaceProperty


**/

#include <sstream>
#include <string>
#include <iomanip>

class G4OpticalSurface ; 

struct U4OpticalSurface
{
    static std::string Desc(const G4OpticalSurface* surf) ; 
};

#include "U4SurfaceType.h"
#include "U4OpticalSurfaceModel.h"
#include "U4OpticalSurfaceFinish.h"

std::string U4OpticalSurface::Desc(const G4OpticalSurface* surf)
{
    G4SurfaceType type = surf->GetType();        // dielectric_metal,dielectric_dielectric 
    G4OpticalSurfaceModel model = surf->GetModel();     // glisur,unified,.. 
    G4OpticalSurfaceFinish finish = surf->GetFinish() ; // polished,polishedfrontpainted,polishedbackpainted,ground,...
    G4double polish = surf->GetPolish(); 
    G4double sigma_alpha = surf->GetSigmaAlpha(); 

    std::stringstream ss ; 
    ss << "U4OpticalSurface::Desc"
       << " " <<  std::setw(20) << surf->GetName() 
       << " type:" << U4SurfaceType::Name(type) 
       << " model:" << U4OpticalSurfaceModel::Name(model) 
       << " finish:" << U4OpticalSurfaceFinish::Name(finish)
       ;

    if(model == glisur) ss << " polish:" << polish ; 
    else                ss << " sigma_alpha:" << sigma_alpha ; 

    std::string s = ss.str(); 
    return s ; 
}




