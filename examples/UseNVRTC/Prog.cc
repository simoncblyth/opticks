


#include <exception>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <string>
#include <vector>
#include <nvrtc.h>

#include "Prog.h"

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

#define NVRTC_CHECK_ERROR( func )                                  \
  do {                                                             \
    nvrtcResult code = func;                                       \
    if( code != NVRTC_SUCCESS )                                    \
      throw std::runtime_error( "ERROR: " __FILE__ "(" LINE_STR "): " + std::string( nvrtcGetErrorString( code ) ) ); \
  } while( 0 )


Prog::Prog(const char* name_, const char* source_, int numHeaders_, const char** headers_, const char** includeNames_ )
    :
    name(strdup(name_)),
    source(strdup(source_)),
    numHeaders(numHeaders_),
    headers(headers_), 
    includeNames(includeNames_),
    prog(0),
    logSize(0),
    log(nullptr),
    ptxSize(0),
    ptx(nullptr)
{
    init(); 
} 


/**
https://docs.nvidia.com/cuda/nvrtc/index.html

https://github.com/NVIDIA/jitify/blob/master/jitify.hpp#L1625

https://stackoverflow.com/questions/40087364/how-do-you-include-standard-cuda-libraries-to-link-with-nvrtc-code





numHeaders
    Number of headers used. 
    numHeaders must be greater than or equal to 0.

headers
    Sources of the headers. 
    headers can be NULL when numHeaders is 0.

includeNames
    Name of each header by which they can be included in the CUDA program source. 
    includeNames can be NULL when numHeaders is 0.

If there are any #include directives, the contents of the files that are
#include'd can be passed as elements of headers, and their names as elements of
includeNames. 

For example::
 
     #include <foo.h> 
     #include <bar.h> 

would require:

numHeaders
   2 

headers
   { "<contents of foo.h>", "<contents of bar.h>" } 

includeNames

   { "foo.h", "bar.h" } 

Alternatively, the compile option -I can be used if the header is guaranteed to exist in the file system at runtime.

**/

void Prog::init()
{
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, source, name, numHeaders, headers, includeNames )) ;
}

void Prog::compile(int numOptions, const char** opts )
{
    NVRTC_CHECK_ERROR( nvrtcCompileProgram(prog, numOptions, opts)) ;

    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize(prog, &logSize) );

    log = new char[logSize];

    NVRTC_CHECK_ERROR( nvrtcGetProgramLog(prog, log) );

    NVRTC_CHECK_ERROR( nvrtcGetPTXSize(prog, &ptxSize));

    ptx = new char[ptxSize];

    NVRTC_CHECK_ERROR( nvrtcGetPTX(prog, ptx) );

    NVRTC_CHECK_ERROR( nvrtcDestroyProgram(&prog) );
}

void Prog::dump() const 
{
    std::cout 
        << "[log size " << logSize 
        << std::endl 
        << log 
        << std::endl 
        << "]log" 
        << std::endl
        ;
 
    std::cout 
        << "[ptx size " << ptxSize
        << std::endl 
        << ptx 
        << std::endl
        << "]ptx" 
        << std::endl
        ;
}

