#pragma once
/**
U4Simtrace.h
==============



**/

#include <iostream>
#include "U4Navigator.h"
#include "SEvt.hh"
#include "U4Tree.h"

struct U4Simtrace
{
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
    std::cout << "[U4Simtrace::EndOfRunAction\n" ;
    Scan(tree);
    std::cout << "]U4Simtrace::EndOfRunAction\n" ;
}

inline void U4Simtrace::Scan(const U4Tree* tree)
{
    int eventID = 998 ;

    SEvt* evt = SEvt::CreateSimtraceEvent();
    evt->beginOfEvent(eventID);

    int num_simtrace = int(evt->simtrace.size()) ;

    if(SEvt::SIMTRACE) std::cout
        << "[U4Simtrace::Scan"
        << " num_simtrace " << num_simtrace
        << " evt.desc "
        << std::endl
        << evt->desc()
        << std::endl
        ;


    U4Navigator nav(tree);
    for(int i=0 ; i < num_simtrace ; i++)
    {
        quad4& p = evt->simtrace[i] ;
        nav.simtrace(p);
        if( i % 10000 == 0 ) std::cout << nav.isect.desc() << "\n" ;
    }
    std::cout << nav.stats.desc() << "\n" ;



    evt->gather();        // follow QSim::simulate
    evt->topfold->concat();
    evt->topfold->clear_subfold();

    evt->endOfEvent(eventID);

    if(SEvt::SIMTRACE) std::cout
        << "]U4Simtrace::Scan"
        << " num_simtrace " << num_simtrace
        << " evt.desc "
        << std::endl
        << evt->desc()
        << std::endl
        ;

}


