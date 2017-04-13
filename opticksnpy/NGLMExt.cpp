#include "SDigest.hh"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include <glm/gtx/component_wise.hpp> 
#include <glm/gtx/matrix_operation.hpp>


glm::mat4 nglmext::invert_tr( const glm::mat4& tr )
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

glm::mat4* nglmext::invert_tr( const glm::mat4* tr )
{
    if(tr == NULL) return NULL ; 
    return new glm::mat4( invert_tr(*tr) );
}

glm::mat4 nglmext::average_to_inverse_transpose( const glm::mat4& m )
{
    glm::mat4 it = glm::inverse(glm::transpose(m)) ;
    return (m + it)/2.f ;
}

ndeco nglmext::polar_decomposition( const glm::mat4& trs, bool verbose )
{
    ndeco d ; 

    d.t = glm::translate(glm::mat4(1.f), glm::vec3(trs[3])) ; 
    d.it = glm::translate(glm::mat4(1.f), -glm::vec3(trs[3])) ; 

    d.rs = glm::mat4(glm::mat3(trs)) ;

    glm::mat4 prev = d.rs ; 
    glm::mat4 next ; 

    float diff ; 
    int count(0) ; 
    do {
        next = average_to_inverse_transpose( prev ) ;
        diff = compDiff(prev, next) ;
        prev = next ; 

        if(verbose)
        std::cout << "polar_decomposition"
                  << " diff " << diff 
                  << " count " << count 
                  << std::endl ; 

    } while( ++count < 100 && diff > 0.0001f ); 

    d.r = next ;
    d.ir = glm::transpose(d.r) ;
    d.s = glm::transpose(d.r) * d.rs ;   //  input rs matrix M,  S = R^-1 M

    glm::vec4 isca(0,0,0,1) ; 
    for(unsigned i=0 ; i < 3 ; i++) isca[i] = 1.f/d.s[i][i] ; 
    
    d.is = glm::diagonal4x4(isca);

    d.isirit = d.is * d.ir * d.it ; 
    d.trs = d.t * d.r * d.s  ; 

    return d ; 
} 



glm::mat4 nglmext::invert_trs( const glm::mat4& trs )
{
    /**
    Input transforms are TRS (scale first, then rotate, then translate)::

          T*R*S*v

    invert by dis-membering trs into rs and t by inspection 
    then extract the r by polar decomposition, ie by 
    iteratively averaging with the inverse transpose until 
    the iteration stops changing much ... at which point
    are left with the rotation portion

    Then separately transpose the rotation,
    negate the translation and reciprocate the scaling 
    and multiply in reverse order

          IS*IR*IT

    The result should be close to directly taking 
    the inverse and has advantage that it tests the form 
    of the transform.
 
    **/

    ndeco d = polar_decomposition( trs ) ;
    glm::mat4 isirit = d.isirit ; 
    glm::mat4 i_trs = glm::inverse( trs ) ; 

    float diff = compDiff(isirit, i_trs );
    assert( diff < 1e-4 );

    return isirit ; 
}



float nglmext::compDiff(const glm::mat4& a , const glm::mat4& b )
{
    // maximum absolute componentwise difference 

    glm::mat4 amb = a - b ; 

    glm::mat4 aamb ; 
    for(unsigned i=0 ; i < 4 ; i++) aamb[i] = glm::abs(amb[i]) ; 

    glm::vec4 colmax ; 
    for(unsigned i=0 ; i < 4 ; i++) colmax[i] = glm::compMax(aamb[i]) ;

    return glm::compMax(colmax) ; 
}



glm::mat4 nglmext::make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal )
{
    glm::mat4 mat(1.f) ;

    float angle_radians = glm::pi<float>()*axis_angle.w/180.f ; 

    for(unsigned i=0 ; i < order.length() ; i++)
    {
        switch(order[i])
        {
           case 's': mat = glm::scale(mat, scal)         ; break ; 
           case 'r': mat = glm::rotate(mat, angle_radians, glm::vec3(axis_angle)) ; break ; 
           case 't': mat = glm::translate(mat, tlat )    ; break ; 
        }
    }
    // for fourth column translation unmodified the "t" must come last, ie "trs"
    return mat  ; 
}

glm::mat4 nglmext::make_transform(const std::string& order)
{
    glm::vec3 tla(0,0,100) ; 
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(1,1,1) ; 

    return make_transform(order, tla, rot, sca );
}






std::string nmat4pair::digest()
{
    return SDigest::digest( (void*)this, sizeof(nmat4pair) );
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





std::ostream& operator<< (std::ostream& out, const glm::ivec3& v) 
{
    out << "{" 
        << " " << std::setw(4) << v.x 
        << " " << std::setw(4) << v.y 
        << " " << std::setw(4) << v.z
        << "}";
    return out;
}




std::ostream& operator<< (std::ostream& out, const glm::vec3& v) 
{
    out << "{" 
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.x 
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.y
        << " " << std::fixed << std::setprecision(4) << std::setw(9) << v.z 
        << "}";

    return out;
}



std::ostream& operator<< (std::ostream& out, const glm::vec4& v) 
{
    out << "{" 
        << " " << std::setprecision(2) << std::setw(7) << v.x 
        << " " << std::setprecision(2) << std::setw(7) << v.y
        << " " << std::setprecision(2) << std::setw(7) << v.z 
        << " " << std::setprecision(2) << std::setw(7) << v.w 
        << "}";
    return out;
}

std::ostream& operator<< (std::ostream& out, const glm::mat4& v) 
{
    out << "( "
        << " " << v[0]
        << " " << v[1]
        << " " << v[2]
        << " " << v[3]
        << " )"
        ; 

    return out;
}


std::ostream& operator<< (std::ostream& out, const glm::mat3& v) 
{
    out << "( "
        << " " << v[0]
        << " " << v[1]
        << " " << v[2]
        << " )"
        ; 

    return out;
}








