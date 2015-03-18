#include "LaunchCommon.hh"

#include <string.h>   
#include <stdlib.h>   
#include <sys/stat.h>   
#include <errno.h>

int getenvvar(const char* name, int def)
{
   int ivar = def ; 
   char* evar = getenv(name);
   if (evar!=NULL) ivar = atoi(evar);
   return ivar ;
}


int mkdirp(const char* _path, int mode) 
{
    // directory tree creation by swapping slashes for end of string '\0'
    // then restoring the slash 
    //  
    // NB when given a file path to be created this does NOT do the
    // the right thing : it creates a directory named like intended filepath 
    //  
    //  http://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
    //  printf("_path %s \n", _path);

    char* path = strdup(_path);
    char* p = path ;
    int rc = 0 ; 

    while (*p != '\0') 
    {   
        p++;
        while(*p != '\0' && *p != '/') p++;

        char v = *p;  // hold on to the '/'
        *p = '\0';
    
        //printf("path [%s] \n", path);

        rc = mkdir(path, mode);

        if(rc != 0 && errno != EEXIST) 
        {   
            *p = v;
            rc = 1;
            break ;
        }   
        *p = v;
    }   

    free(path); 
    return rc; 
}




