#include "Opticks.hh"

#include "BRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv, char** /*envp*/)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ;
    OKCORE_LOG__ ;

    Opticks ok(argc, argv) ;
    ok.configure();
    ok.Summary();

    if(argc > 1 && strlen(argv[1]) > 0)
    {
        ok.saveResources(argv[1]);
    }
    else
    {
        ok.saveResources();
    }

    return 0 ; 
}
