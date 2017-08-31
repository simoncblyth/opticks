#pragma once

#include "OGLRAP_API_EXPORT.hh"

struct OGLRAP_API ProgLog
{
    static const char* NO_FRAGMENT_SHADER ; 
    enum { MAX_LENGTH = 2048 } ;

    ProgLog(int id_);
    bool is_no_frag_shader() const ;

    void dump(const char* msg);

    int   id ; 
    int   length ; 
    char  log[MAX_LENGTH];
};


