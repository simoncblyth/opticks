#pragma once
/**
U4Simtrace.h
==============



**/

#include <iostream>
#include "U4Navigator.h"

struct U4Simtrace
{
    static void EndOfRunAction(); 
};


inline void U4Simtrace::EndOfRunAction()
{
    std::cout 
        << "U4Simtrace::EndOfRunAction"
        << std::endl
        ;

    U4Navigator::Check(); 
}


