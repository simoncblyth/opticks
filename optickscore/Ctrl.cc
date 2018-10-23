#include <cassert>
#include <sstream>
#include "Ctrl.hh"


Ctrl::Ctrl(float* fptr, unsigned n) 
{
    assert( n == 4 ); 
    for(unsigned j=0 ; j < n ; j++ ) fc.f[j] = *(fptr+j) ; 

    for(int i=0 ; i < 8 ; i++)
    {
        char* p = fc.c + i*2 ;
        if(*p == 0) continue ; 
        std::string cmd(p, 2) ;
        cmds.push_back(cmd); 
    }
}  


std::string Ctrl::getCommands() const 
{
    std::stringstream ss ; 
    unsigned num = cmds.size() ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        ss 
           << cmds[i] 
           << ( i < num - 1 ? "," : "")
        ; 
    }
    return ss.str();
}



