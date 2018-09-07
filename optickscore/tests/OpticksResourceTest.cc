// op --resource
// op --j1707 --resource

#include <cstring>
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OPTICKS_LOG.hh"


void dumpenv_0(char** envp, const char* prefix)
{
   // http://stackoverflow.com/questions/2085302/printing-all-environment-variables-in-c-c
    char** env;

    for (env = envp; *env != 0; env++)
    {
       char* thisEnv = *env;

       if( strlen(thisEnv) > strlen(prefix) && strncmp( thisEnv, prefix, strlen(prefix) ) == 0 ) 
       printf("%s\n", thisEnv);    
    }
}

void dumpenv_1(char** envp)
{
    while(*envp) printf("%s\n",*envp++);
}



int main(int argc, char** argv, char** envp)
{
    dumpenv_0( envp, "OPTICKS_" ) ; 

    OPTICKS_LOG(argc, argv);

    /*
    No longer needed, to make sensitive to OPTICKS_KEY envvar use "--envkey" option 

    const char* demokey = "OKX4Test.X4PhysicalVolume.World0xc15cfc0_PV.0dce832a26eb41b58a000497a3127cb8" ; 
    const char* ukey = NULL ; 
    if( argc > 1 && strcmp(argv[1], "--demokey") == 0) ukey = demokey ;   
    Opticks::SetKey(ukey);  // <-- using NULL makes sensitive to OPTICKS_KEY envvar for debugging
    */


    Opticks ok(argc, argv) ;
    ok.dumpResource(); 

    return 0 ; 
}
