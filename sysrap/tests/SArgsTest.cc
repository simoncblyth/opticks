#include <cassert>
#include "SArgs.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* extra = "--compute --nopropagate --tracer" ;

    SArgs* sa = new SArgs(argc, argv, extra );

    std::cout << " sa->argc " << sa->argc << std::endl ; 


    sa->dump();




    assert(sa->hasArg("--compute"));
    assert(sa->hasArg("--nopropagate"));
    assert(sa->hasArg("--tracer"));

    std::string e = "--hello" ;  
    std::string f = "hello" ;  
    assert( SArgs::starts_with(e,"--") == true ) ;
    assert( SArgs::starts_with(f,"--") == false ) ;

    return 0 ; 
}
/*
The below should be deduped:

    SArgsTest  --tracer --compute
    SArgsTest  --tracer --compute --nopropagate   

Deduping only has effect between the argforced "extra" additions
and the ordinary args, it does not dedupe the standard args.

*/
