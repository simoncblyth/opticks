#pragma once

#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <errno.h> 
#include "dirent.h"

struct sdirectory
{
   static int MakeDirs(const char* dirpath, int mode );  
};

/**
sdirectory::MakeDirs
----------------------

While loop is needed because mkdir can only create one 
level of directory at once. 
This follows NPU.hh U::MakeDirs

**/

inline int sdirectory::MakeDirs( const char* dirpath_, int mode_ )
{
    mode_t default_mode = S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH ;
    mode_t mode = mode_ == 0 ? default_mode : mode_ ;

    char* dirpath = strdup(dirpath_);
    char* p = dirpath ;
    int rc = 0 ;

    while (*p != '\0' && rc == 0)
    {
        p++;   // advance past leading character, probably slash, and subsequent slashes the next line gets to  
        while(*p != '\0' && *p != '/') p++;  // advance p until subsequent slash 
        char v = *p;                         // store the slash      
        *p = '\0' ;                          // replace slash with string terminator
        //printf("%s\n", path );                   
        rc = mkdir(dirpath, mode) == -1 && errno != EEXIST ? 1 : 0 ;  // set rc non-zero for mkdir errors other than exists already  
        *p = v;                              // put back the slash  
    }   
    free(dirpath);
    return rc ; 
}


