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
    SEvt* evt = SEvt::CreateSimtraceEvent();  
    bool dump = false ; 
    for(int i=0 ; i < int(evt->simtrace.size()) ; i++)
    {
        quad4& p = evt->simtrace[i] ;
        U4Navigator::Simtrace(p, dump);
    }
    evt->save(); 
}


