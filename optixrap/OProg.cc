#include <cstring>
#include <cstdio>
#include "OProg.hh"

OProg::OProg(char type_, unsigned int index_, const char* filename_, const char* progname_)  
         :
         type(type_),
         index(index_),
         filename(strdup(filename_)),
         progname(strdup(progname_)),
         _description(NULL)
{
}

const char* OProg::description()
{
    if(!_description)
    {
        char desc[128];
        snprintf(desc, 128, "OProg %c %u %s %s ", type, index, filename, progname );
        _description = strdup(desc);
    }
    return _description ;
}

