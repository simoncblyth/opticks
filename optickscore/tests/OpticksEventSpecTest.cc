// TEST=OpticksEventSpecTest om-t

#include "OpticksEventSpec.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    OpticksEventSpec s0("typ","tag","det") ; 
    s0.Summary("s0 (no cat)");

    OpticksEventSpec s1("typ","tag","det", "cat") ; 
    s1.Summary("s1 (with cat)");

    return 0 ; 
}
