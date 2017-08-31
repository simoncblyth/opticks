#include <cstring>
#include <GL/glew.h>

#include "PLOG.hh"
#include "ProgLog.hh"

const char* ProgLog::NO_FRAGMENT_SHADER = "Validation Failed: Program does not contain fragment shader. Results will be undefined.\n" ;

ProgLog::ProgLog(int id_) : id(id_), length(0) 
{
    glGetProgramInfoLog (id, MAX_LENGTH, &length, log);
}

bool ProgLog::is_no_frag_shader() const 
{
    return strcmp(NO_FRAGMENT_SHADER, log) == 0 ;
}

void ProgLog::dump(const char* msg)
{
    LOG(info) << msg ; 
    printf ("ProgLog::dump id %u:\n[%s]", id, log);
}



