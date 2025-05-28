#pragma once
/**
U4Simtrace.h
==============



**/

#include <iostream>
#include "U4Navigator.h"
#include "SEvt.hh"

struct U4Simtrace
{
    static void EndOfRunAction();
    static void Scan();
};

inline void U4Simtrace::EndOfRunAction()
{
    std::cout << "[U4Simtrace::EndOfRunAction\n" ;
    Scan();
    std::cout << "]U4Simtrace::EndOfRunAction\n" ;
}

inline void U4Simtrace::Scan()
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

    bool dump = false ;
    for(int i=0 ; i < num_simtrace ; i++)
    {
        quad4& p = evt->simtrace[i] ;
        U4Navigator::Simtrace(p, dump);
    }


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


