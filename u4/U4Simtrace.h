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
    std::cout 
        << "[ U4Simtrace::EndOfRunAction"
        << std::endl
        ;

    Scan(); 

    std::cout 
        << "] U4Simtrace::EndOfRunAction"
        << std::endl
        ;
}

inline void U4Simtrace::Scan()
{
    int eventID = 998 ; 

    SEvt* evt = SEvt::CreateSimtraceEvent();  
    evt->beginOfEvent(eventID); 

    int num_simtrace = int(evt->simtrace.size()) ;

    std::cout
        << "U4Simtrace::Scan"
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
    evt->endOfEvent(eventID);
}


