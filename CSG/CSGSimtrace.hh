#pragma once

#include "plog/Severity.h"
#include "ssys.h"
#include "sframe.h"

struct CSGFoundry ; 
struct SEvt ; 
struct SSim ; 
struct CSGQuery ; 
struct CSGDraw ; 


#include "CSG_API_EXPORT.hh"

struct CSG_API CSGSimtrace
{
    static const plog::Severity LEVEL ; 
    static int Preinit(); 

    int prc ; 
    const char* geom ;
    SSim* sim ; 
    const CSGFoundry* fd ; 
    SEvt* evt ; 
    sframe frame ;
    CSGQuery* q ; 
    CSGDraw* d ; 

    CSGSimtrace();  
    void simtrace();
    void saveEvent();  
}; 




