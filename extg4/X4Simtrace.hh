#pragma once
/**
X4Simtrace : aiming to replace X4Intersect
=============================================

**/
#include <cstring>
#include "plog/Severity.h"
#include "sframe.h"
#include "ssys.h"

class G4VSolid ; 
struct SEvt ; 

#include "X4_API_EXPORT.hh"

struct X4_API X4Simtrace
{
    static const plog::Severity LEVEL ; 
    static void Scan(const G4VSolid* solid); 

    const G4VSolid* solid ; 
    SEvt* evt ; 
    sframe frame ;  

    X4Simtrace(); 
    void setSolid(const G4VSolid* solid); 
    void simtrace(); 
    void saveEvent(); 
};

inline X4Simtrace::X4Simtrace()
    :
    solid(nullptr), 
    evt(nullptr)
{
}

