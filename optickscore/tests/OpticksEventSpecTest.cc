#include "OpticksEventSpec.hh"

#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG_ ; 


    OpticksEventSpec s0("typ","tag","det") ; 
    s0.Summary("s0 (no cat)");

    OpticksEventSpec s1("typ","tag","det", "cat") ; 
    s1.Summary("s1 (with cat)");

    return 0 ; 
}
