#include "Nuv.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    unsigned s = 0 ; 
    unsigned u = 1 ; 
    unsigned v = 2 ; 
    unsigned nu = 10 ; 
    unsigned nv = 20 ; 
    unsigned pr = 0 ; 

    nuv p = make_uv(s,u,v,nu,nv, pr);

    assert( p.s() == s ) ; 
    assert( p.u() == u ) ; 
    assert( p.v() == v ) ; 
    assert( p.nu() == nu ); 
    assert( p.nv() == nv ); 
    assert( p.nv() == nv ); 
    assert( p.p() == pr ); 

    LOG(info) << p.desc() ; 


    return 0 ; 
}
