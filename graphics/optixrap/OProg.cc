#include "OProg.hh"

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

