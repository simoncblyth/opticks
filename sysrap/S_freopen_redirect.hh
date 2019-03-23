// https://stackoverflow.com/questions/1908687/how-to-redirect-the-output-back-to-the-screen-after-freopenout-txt-a-stdo

/**
S_freopen_redirect
===================

struct for file descriptor gymnastics. 

Used from :doc:`/optixrap/OContext` to redirect OptiX kernel debug stdout to a file.

**/


#include <cstdio>
#include <cstring>
#include <unistd.h>

struct S_freopen_redirect
{
    FILE*       _old    ;
    int         _old_fd ; 
    const char* _path    ; 

    S_freopen_redirect( FILE* curr, const char* path )
        :
        _old(curr),
        _old_fd(dup(fileno(curr))),
        _path(strdup(path))
    {
        freopen( _path, "w", curr );
    }

    ~S_freopen_redirect()
    {
        fclose(_old);
        FILE *fp2 = fdopen(_old_fd, "w");
        *_old = *fp2;  // Unreliable!
    }                      
};

