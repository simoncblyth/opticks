// TEST=OpticksEventSpecTest om-t

#include "OpticksEventSpec.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* tag = "1" ; 

    OpticksEventSpec s0("pfx", "typ", tag, "det") ; 
    s0.Summary("s0 (no cat)");

    OpticksEventSpec s1("pfx", "typ", tag, "det", "cat") ; 
    s1.Summary("s1 (with cat)");

    return 0 ; 
}
