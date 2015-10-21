#pragma once
#include <cstring>
#include <cstdio>

struct OProg {
    OProg(char type_, unsigned int index_, const char* filename_, const char* progname_)  
         :
         type(type_),
         index(index_),
         filename(strdup(filename_)),
         progname(strdup(progname_)),
         _description(NULL)
    {
    }

    const char* description();

    char type ;
    unsigned int index ; 
    const char* filename ;  
    const char* progname ;  
    const char* _description ; 
};


