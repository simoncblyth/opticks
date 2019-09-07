/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>
#include <sstream>
#include <iomanip>

#include "G4OpticalSurface.hh"
#include "COpticalSurface.hh"

COpticalSurface::COpticalSurface(G4OpticalSurface* os)
    :
    m_os(os)
{
}


const char* COpticalSurface::polished_ = "polished" ;
const char* COpticalSurface::polishedfrontpainted_ = "polishedfrontpainted" ;
const char* COpticalSurface::polishedbackpainted_  = "polishedbackpainted" ;
const char* COpticalSurface::ground_ = "ground" ;
const char* COpticalSurface::groundfrontpainted_ = "groundfrontpainted" ;
const char* COpticalSurface::groundbackpainted_  = "groundbackpainted" ;
const char* COpticalSurface::Finish( G4OpticalSurfaceFinish finish)
{
    const char* s = NULL ;
    switch(finish)
    {
       case polished             : s =  polished_             ; break ;
       case polishedfrontpainted : s =  polishedfrontpainted_ ; break ;
       case polishedbackpainted  : s =  polishedbackpainted_  ; break ;
       case ground               : s =  ground_               ; break ;
       case groundfrontpainted   : s =  groundfrontpainted_   ; break ;
       case groundbackpainted    : s =  groundbackpainted_    ; break ;
       default: assert(0 && "unexpected optical surface finish") ; break ;
    }
    return s ;
}

const char* COpticalSurface::dielectric_dielectric_ = "dielectric_dielectric" ;
const char* COpticalSurface::dielectric_metal_      = "dielectric_metal" ;
const char* COpticalSurface::Type( G4SurfaceType type)
{
    const char* s = NULL ;
    switch(type)
    {
       case dielectric_metal     : s =  dielectric_metal_      ; break ;
       case dielectric_dielectric: s =  dielectric_dielectric_ ; break ;
       default: assert(0 && "unexpected optical surface type") ; break ;
    }
    return s ;
}


const char* COpticalSurface::glisur_  = "glisur" ;
const char* COpticalSurface::unified_ = "unified" ;
const char* COpticalSurface::Model( G4OpticalSurfaceModel model )
{
    const char* s = NULL ;
    switch(model)
    {
       case glisur  : s =  glisur_  ; break ;
       case unified : s =  unified_ ; break ;
       default: assert(0 && "unexpected optical surface model") ; break ;
    }
    return s ;


}





std::string COpticalSurface::brief()
{
    return Brief(m_os);
}

std::string COpticalSurface::Brief(G4OpticalSurface* os)
{
    std::stringstream ss ; 

    G4OpticalSurfaceModel model = os->GetModel();
    G4SurfaceType type = os->GetType() ;
    G4OpticalSurfaceFinish finish = os->GetFinish();

    ss << std::setw(30) << os->GetName()
       << " model " << std::setw(30) << Model(model)
       << " type " << std::setw(30) << Type(type)
       << " finish " << std::setw(30) << Finish(finish)
       ;


    return ss.str();
}










   
