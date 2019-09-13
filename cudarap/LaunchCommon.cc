/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "LaunchCommon.hh"

#include <cstring>   
#include <cstdlib>   
#include <sys/stat.h>   
#include <errno.h>


/*
int getenvvar(const char* name, int def)
{
   int ivar = def ; 
   char* evar = getenv(name);
   if (evar!=NULL) ivar = atoi(evar);
   return ivar ;
}
*/

/**
mkdirp
----------

directory tree creation by swapping slashes for end of string '\0'
then restoring the slash 

NB when given a file path to be created this does NOT do the
the right thing : it creates a directory named like intended filepath 

http://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
printf("_path %s \n", _path);

**/


int mkdirp(const char* _path, int mode) 
{
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




