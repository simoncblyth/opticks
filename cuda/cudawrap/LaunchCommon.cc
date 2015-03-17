#include "LaunchCommon.hh"

#include <stdlib.h>   

int getenvvar(const char* name, int def)
{
   int ivar = def ; 
   char* evar = getenv(name);
   if (evar!=NULL) ivar = atoi(evar);
   return ivar ;
}


