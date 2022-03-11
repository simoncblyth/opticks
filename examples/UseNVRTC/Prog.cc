


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





* https://forums.developer.nvidia.com/t/include-header-in-nvrtc/56715

Robert_Crovella January 5, 2018, 10:43pm #14 A description of how to use these
parameters is contained here:

http://docs.nvidia.com/cuda/nvrtc/index.html#basic-usage

You are still using them incorrectly.

The nvrtc mechanism gives 2 methods by which an include file can be
incorporated into your program:

By passing the entire contents of the include file to nvrtcCreateProgram By
passing just the path of the include file to nvrtcCompileProgram The second
method is probably easier. The first method is provided in case you cannot be
certain that a particular include file will be available or findable at the
place you expect in the filesystem, when the program is being compiled at
runtime.

The first method is not really outlined anywhere (although it is offered via
jitify). An example of the second method is available in the CUDA sample codes,
in the file /usr/local/cuda/samples/common/inc/nvrtc_helper.h Study the
function compileFileToPTX in that helper file, and note how the
cooperative_groups.h header file is passed to the nvrtcCompileProgram function.

/usr/local/cuda/samples/common/inc/nvrtc_helper.h::

     42 
     43     int numCompileOptions = 0;
     44 
     45     char *compileParams[1];
     46 
     47     if (requiresCGheaders)
     48     {
     49         std::string compileOptions;
     50         char *HeaderNames = "cooperative_groups.h";
     51 
     52         compileOptions = "--include-path=";
     53 
     54         std::string path = sdkFindFilePath(HeaderNames, argv[0]);
     55         if (!path.empty())
     56         {
     57             std::size_t found = path.find(HeaderNames);
     58             path.erase(found);
     59         }
     60         else
     61         {
     62             printf("\nCooperativeGroups headers not found, please install it in %s sample directory..\n Exiting..\n", argv[0]);
     63         }
     64         compileOptions += path.c_str();
     65         compileParams[0] = (char *) malloc(sizeof(char)* (compileOptions.length() + 1));
     66         strcpy(compileParams[0], compileOptions.c_str());
     67         numCompileOptions++;
     68     }
     69 
     70     // compile
     71     nvrtcProgram prog;
     72     NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, memBlock,
     73                                                      filename, 0, NULL, NULL));
     74 
     75     nvrtcResult res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);
     76 
        


If you want to use the first method, don’t pass the path to the file in the
header_sources, pass the complete contents of the include file in
header_sources. This is the proximal reason for the error you are getting. The
compiler is attempting to interpret /home/martin/ as the C/C++ code contents of
your header file.




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

