// name=mkdirp ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name
// https://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <errno.h>

/**
//                        0700      040        010       04        01 

In [13]: print(oct(0o700 | 0o40 | 0o10 | 0o4 | 0o01))                                                                                                                                                     
0o755

**/

int mkdirp(const char* path, int mode_ = 0  );


int mkdirp(const char* path_, int mode_ ) 
{
    mode_t default_mode = S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH ; 
    mode_t mode = mode_ == 0 ? default_mode : mode_ ;  

    char* path = strdup(path_);
    char* p = path ;  
    int rc = 0 ; 

    while (*p != '\0' && rc == 0) 
    {
        p++;                                 // advance past leading character, probably slash, and subsequent slashes the next line gets to  
        while(*p != '\0' && *p != '/') p++;  // advance p until subsequent slash 
        char v = *p;                         // store the slash      
        *p = '\0' ;                          // replace slash with string terminator
        printf("%s\n", path ); 
        rc = mkdir(path, mode) == -1 && errno != EEXIST ? 1 : 0 ;  // set rc non-zero for mkdir errors other than exists already  
        *p = v;                              // put back the slash 
    }
    free(path); 
    return rc ;
}

int main(int argc, char** argv)
{
    const char* path = "/tmp/red/green/blue" ; 
    int mode = 0777 ; 
    int rc = mkdirp(path, mode); 
    return rc ; 
}



