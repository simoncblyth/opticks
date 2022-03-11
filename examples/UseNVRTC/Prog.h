#pragma once

//#include <nvrtc.h>
struct _nvrtcProgram ; 

struct Prog
{
    const char*  name ; 
    const char*  source ; 
    int          numHeaders ; 
    const char** headers ;      // content of header files 
    // HMM: what about headers that include other headers
    const char** includeNames ; 

    //nvrtcProgram prog ; 
    _nvrtcProgram* prog ; 

    size_t logSize;
    char*  log ; 
    size_t ptxSize;
    char*  ptx ;

    Prog(const char* name, const char* source, int numHeaders, const char** headers, const char** includeNames ); 
    void init(); 
    void compile(int numOptions, const char** opts ); 
    void dump() const ; 
};


