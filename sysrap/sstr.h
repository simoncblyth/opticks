#pragma once

struct sstr
{
    static const char* TrimTrailing(const char* s);
};

#include <cstring>

inline const char* sstr::TrimTrailing(const char* s)
{
    char* p = strdup(s); 
    char* e = p + strlen(p) - 1 ;  
    while(e > p && ( *e == ' ' || *e == '\n' )) e-- ;
    e[1] = '\0' ;
    return p ;  
}

