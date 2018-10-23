#pragma once

#include <string>
#include <vector>
#include "OKCORE_API_EXPORT.hh"




struct OKCORE_API Ctrl 
{
    union u_f4_c16
    {
         float f[4] ;
         char  c[16] ;
    };

    Ctrl(float* ptr, unsigned n=4); 
    std::string getCommands() const ; 

    u_f4_c16 fc ; 

    std::vector<std::string> cmds ;

};




 
