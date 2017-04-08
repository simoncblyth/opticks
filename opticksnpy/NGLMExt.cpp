#include "NGLMExt.hpp"
#include "GLMFormat.hpp"


glm::mat4 invert_tr( const glm::mat4& tr )
{
    /**
       input transforms are rotation first then translation :  T*R*v
     
       invert by dis-membering tr into r and t by inspection and separately  
       transpose the rotation and negate the translation then 
       multiply in reverse order

               IR*IT 
    **/

    glm::mat4 ir = glm::transpose(glm::mat4(glm::mat3(tr)));
    glm::mat4 it = glm::translate(glm::mat4(1.f), -glm::vec3(tr[3])) ; 
    glm::mat4 irit = ir*it ;    // <--- inverse of tr 
    return irit ; 
}

glm::mat4* invert_tr( const glm::mat4* tr )
{
    if(tr == NULL) return NULL ; 
    return new glm::mat4( invert_tr(*tr) );
}





nmat4pair* nmat4pair::product(const std::vector<nmat4pair*>& tt)
{
    unsigned ntt = tt.size();
    if(ntt==0) return NULL ; 
    if(ntt==1) return tt[0] ; 

    glm::mat4 tr(1.0) ; 
    glm::mat4 irit(1.0) ; 

    for(unsigned i=0,j=ntt-1 ; i < ntt ; i++,j-- )
    {
        std::cout << " i " << i << " j " << j << std::endl ; 
 
        const nmat4pair* ti = tt[i] ; 
        const nmat4pair* tj = tt[j] ; 

        tr *= ti->tr ; 
        irit *= tj->irit ;   // guess multiplication ordering 
    }

    // is this the appropriate transform and inverse transform multiplication order ?
    // ... tt order is from the leaf back to the root   

    return new nmat4pair(tr, irit) ; 
}




std::ostream& operator<< (std::ostream& out, const nmat4pair& mp)
{
    out 
       << std::endl 
       << gpresent( "nm4p:tr",   mp.tr ) 
       << std::endl 
       << gpresent( "nm4p:irit", mp.irit )
       << std::endl 
       ; 

    return out;
}


