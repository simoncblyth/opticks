#pragma once

#include "plog/Severity.h"
#include <vector>
#include "sframe.h"

struct CSGFoundry ; 
struct SEvt ; 
struct SSim ; 
struct CSGQuery ; 
struct CSGDraw ; 
struct NP ; 
struct quad4 ; 

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
    const char* outdir ; 

    sframe frame ;
    CSGQuery* q ; 
    CSGDraw* d ; 

    const char* SELECTION ; 
    std::vector<int>* selection ; 
    unsigned num_selection ; 
    NP* selection_simtrace ; 
    quad4* qss ; 

    CSGSimtrace();  
    void init(); 

    int simtrace();
    int simtrace_all();
    int simtrace_selection();

    void saveEvent();  
}; 




