#pragma once
/**
U4Simtrace.h
==============



**/

#include <iostream>
#include "U4Navigator.h"
#include "SEvt.hh"
#include "U4Tree.h"
#include "ssys.h"

struct U4Simtrace
{
    static constexpr const char* U4Simtrace__level = "U4Simtrace__level" ;
    static void EndOfRunAction(const U4Tree* tree);
    static void Scan(const U4Tree* tree);
};

/**
U4Simtrace::EndOfRunAction
---------------------------

Invoked from U4Recorder::EndOfRunAction when configured by envvar

**/


inline void U4Simtrace::EndOfRunAction(const U4Tree* tree)
{
    int level = ssys::getenvint(U4Simtrace__level, 0);
    if(level > 0) std::cout << "[U4Simtrace::EndOfRunAction\n" ;
    Scan(tree);
    if(level > 0) std::cout << "]U4Simtrace::EndOfRunAction\n" ;
}

inline void U4Simtrace::Scan(const U4Tree* tree)
{
    int level = ssys::getenvint(U4Simtrace__level, 0);

    int eventID = 998 ;

    SEvt* evt = SEvt::CreateSimtraceEvent();
    evt->beginOfEvent(eventID);

    int num_simtrace = int(evt->simtrace.size()) ;
    if(level > 0) std::cout << "[U4Simtrace::Scan num_simtrace " << num_simtrace <<  "\n" ;

    U4Navigator nav(tree);

    for(int i=0 ; i < num_simtrace ; i++)
    {
        quad4& p = evt->simtrace[i] ;
        nav.simtrace(p);
        if( level > 1 && i % 1000 == 0 ) std::cout
           << "-U4Simtrace::Scan "
           << " i " << std::setw(10) << i
           << " : " << nav.isect.desc()
           << "\n"
           ;
    }
    if(level > 0) std::cout << "-U4Simtrace::Scan nav.stats.desc\n" << nav.stats.desc() << "\n" ;


    evt->gather();        // follow QSim::simulate
    evt->topfold->concat();
    evt->topfold->clear_subfold();

    evt->endOfEvent(eventID);

    if(level > 0) std::cout << "]U4Simtrace::Scan num_simtrace " << num_simtrace <<  "\n" ;
}


