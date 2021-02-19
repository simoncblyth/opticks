#include "NGLMExt.hpp"
#include "nmat4pair.hpp"
#include "SDigest.hh"
#include "GLMFormat.hpp"
#include "PLOG.hh"

std::string nmat4pair::digest() 
{
    return SDigest::digest( (void*)this, sizeof(nmat4pair) );
}


nmat4pair* nmat4pair::clone()
{
    return new nmat4pair(t,v);
}

nmat4pair* nmat4pair::product(const std::vector<nmat4pair*>& pairs)
{
    unsigned npairs = pairs.size();
    if(npairs==0) return NULL ; 
    if(npairs==1) return pairs[0] ; 

    glm::mat4 t(1.0) ; 
    glm::mat4 v(1.0) ; 

    for(unsigned i=0,j=npairs-1 ; i < npairs ; i++,j-- )
    {
        const nmat4pair* ii = pairs[i] ; 
        const nmat4pair* jj = pairs[j] ; 

        t *= ii->t ; 
        v *= jj->v ; 
    }

    // guessed multiplication ordering 
    // is this the appropriate transform and inverse transform multiplication order ?
    // ... pairs order is from the leaf back to the root   

    return new nmat4pair(t, v) ; 
}




nmat4pair::nmat4pair(const glm::mat4& t_ ) 
     : 
     match(true),
     t(t_),
     v(nglmext::invert_trs(t, match))
{
     if(!match) LOG(error) << " mis-match " ; 

}




std::ostream& operator<< (std::ostream& out, const nmat4pair& pair)
{
    out 
       << std::endl 
       << gpresent( "pair.t",   pair.t ) 
       << std::endl 
       << gpresent( "pair.v", pair.v )
       << std::endl 
       ; 

    return out;
}


