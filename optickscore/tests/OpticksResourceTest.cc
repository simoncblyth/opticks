// op --resource
// op --j1707 --resource

#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OPTICKS_LOG.hh"


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
    OPTICKS_LOG(argc, argv);

    Opticks::SetKey(NULL);  // <-- using NULL makes sensitive to OPTICKS_KEY envvar for debugging

    Opticks ok(argc, argv) ;

    LOG(info) << " ok.getExampleMaterialNames() " << ok.getExampleMaterialNames() ;

    OpticksResource res(&ok) ;  // TODO: remove duplication of envprefix beween both these
    res.Dump();

    return 0 ; 
}
