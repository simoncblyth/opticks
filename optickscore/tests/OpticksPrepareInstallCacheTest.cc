#include "Opticks.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ;
    ok.configure();
    ok.Summary();

    if(argc > 1 && strlen(argv[1]) > 0)
    {
        // canonical usage from opticks-prepare-installcache uses unexpanded argument 
        // '$INSTALLCACHE_DIR/OKC'  expansion is done by BFile.cc
        ok.prepareInstallCache(argv[1]);
    }
    else
    {
        const char* tmp = "$TMP/OKC" ; 
        LOG(info) << "default without argument writes to " << tmp ; 
        ok.prepareInstallCache(tmp);
    }

    return 0 ; 
}


