#pragma once

/**
OProg
======

Holder of the inputs for creating an OptiX program.

**/


#include "OXRAP_API_EXPORT.hh"

struct OXRAP_API OProg {

    OProg(char type_, unsigned int index_, const char* filename_, const char* progname_);

    const char* description();

    char type ;
    unsigned int index ; 
    const char* filename ;  
    const char* progname ;  
    const char* _description ; 
};


