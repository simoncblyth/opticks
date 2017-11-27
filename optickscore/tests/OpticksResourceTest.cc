// op --resource
// op --j1707 --resource

#include "Opticks.hh"
#include "OpticksResource.hh"


#include "OKCORE_LOG.hh"
#include "PLOG.hh"

void dumpenv_0(char** envp)
{
   // http://stackoverflow.com/questions/2085302/printing-all-environment-variables-in-c-c
    char** env;
    for (env = envp; *env != 0; env++)
    {
       char* thisEnv = *env;
       printf("%s\n", thisEnv);    
    }
}


void dumpenv_1(char** envp)
{
    while(*envp) printf("%s\n",*envp++);
}



int main(int argc, char** argv, char** /*envp*/)
{
    PLOG_(argc, argv);
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv) ;

    LOG(info) << " ok.getExampleMaterialNames() " << ok.getExampleMaterialNames() ;

    OpticksResource res(&ok) ;  // TODO: remove duplication of envprefix beween both these
    res.Dump();
    return 0 ; 
}
