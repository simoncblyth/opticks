// name=dirent ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include "stdlib.h"
#include "stdio.h"
#include "dirent.h"

int main(int argc, char** argv)
{
    //const char* path = "c:\\src\\" ; 
    const char* path = "/private/tmp" ; 

    DIR* dir = opendir(path) ;
    if( dir == NULL  )
    {
        perror ("");
        return EXIT_FAILURE;
    }

    struct dirent* ent;

    while ((ent = readdir(dir)) != NULL) 
    {
        printf("%s\n", ent->d_name);
    }

    closedir (dir);


    return 0 ; 
}

