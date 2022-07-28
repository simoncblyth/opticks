#pragma once

#include "scuda.h"
#include "sphoton.h"
#include "U4Hit.h"
#include "U4ThreeVector.h"

struct U4HitConvert
{
    static void FromPhoton(U4Hit& hit, const sphoton& global, const sphoton& local); 
}; 

void U4HitConvert::FromPhoton(U4Hit& hit,  const sphoton& global, const sphoton& local )
{
    U4ThreeVector::FromFloat3( hit.global_position,      global.pos ); 
    U4ThreeVector::FromFloat3( hit.global_direction,     global.mom ); 
    U4ThreeVector::FromFloat3( hit.global_polarization,  global.pol ); 

    hit.time = double(global.time) ; 
    hit.weight = 1. ; 
    hit.wavelength = double(global.wavelength); 

    U4ThreeVector::FromFloat3( hit.local_position,      local.pos ); 
    U4ThreeVector::FromFloat3( hit.local_direction,     local.mom ); 
    U4ThreeVector::FromFloat3( hit.local_polarization,  local.pol ); 
}


