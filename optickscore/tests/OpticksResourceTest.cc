// op --resource

#include "Opticks.hh"
#include "OpticksResource.hh"

int main(int argc, char** argv, char** envp)
{
   // http://stackoverflow.com/questions/2085302/printing-all-environment-variables-in-c-c
    char** env;
    for (env = envp; *env != 0; env++)
    {
       char* thisEnv = *env;
       printf("%s\n", thisEnv);    
    }

    Opticks op(argc, argv) ;
    OpticksResource res(&op) ;  // TODO: remove duplication of envprefix beween both these
    res.Dump();
    return 0 ; 
}
