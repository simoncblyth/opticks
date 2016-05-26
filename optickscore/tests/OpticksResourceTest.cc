// op --resource

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

    OpticksResource res ;
    res.Dump();
    return 0 ; 
}
