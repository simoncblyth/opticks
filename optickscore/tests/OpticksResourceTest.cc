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

    Opticks ok(argc, argv) ;
    ok.configure();
    ok.dumpResource(); 

    return 0 ; 
}
