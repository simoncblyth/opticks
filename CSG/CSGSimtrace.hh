#pragma once
/**
CSGSimtrace.hh : Geometry 2D Cross Sections
==============================================

Canonical usage from tests/CSGSimtraceTest.cc

The heart of this is CSGQuery on CPU intersect functionality using the csg headers


This is a very low level simtrace test that does not use
gensteps, which causes SEvt to issue ignorable warnings.
TODO: avoid the runtime warning from SEvt::addGenstep


**/

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
    SEvt* sev ;
    const char* outdir ;

    sframe frame ;
    CSGQuery* q ;
    CSGDraw* d ;

    const char* SELECTION ;
    std::vector<int>* selection ;
    int num_selection ;
    NP* selection_simtrace ;
    quad4* qss ;

    CSGSimtrace();
    void init();

    int simtrace();
    int simtrace_all();
    int simtrace_selection();

};




