#include <array>
#include <iterator>

#include "SDigest.hh"
#include "NPY.hpp"

#include "NPlane.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include <glm/gtx/component_wise.hpp> 
#include <glm/gtx/matrix_operation.hpp>


#include "PLOG.hh"


void nglmext::copyTransform( std::array<float,16>& dst, const glm::mat4& src )
{
    const float* p = glm::value_ptr(src);
    std::copy(p, p+16, std::begin(dst));
}


std::string nglmext::xform_string( const std::array<float, 16>& xform )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 16 ; i++) 
    {
        bool translation =  i == 12 || i == 13 || i == 14 ; 
        int fwid = translation ? 8 : 6 ;  
        int fprec = translation ? 2 : 3 ; 
        ss << std::setw(fwid) << std::fixed << std::setprecision(fprec) << xform[i] << ' ' ; 
    }
    return ss.str();
}


// Extracts from /usr/local/opticks/externals/yoctogl/yocto-gl/yocto/yocto_gltf.cpp

std::array<float, 16> nglmext::_float4x4_mul( const std::array<float, 16>& a, const std::array<float, 16>& b) 
{
    auto c = std::array<float, 16>();
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            c[j * 4 + i] = 0;
            for (auto k = 0; k < 4; k++)
                c[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
        }
    }
    return c;
}

const std::array<float, 16> nglmext::_identity_float4x4 = {{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};




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

    float diff, diff2  ; 
    int count(0) ; 
    do {
        next = average_to_inverse_transpose( prev ) ;
        diff = compDiff(prev, next) ;
        diff2 = compDiff2(prev, next) ;
        prev = next ; 

        if(verbose)
        std::cout << "polar_decomposition"
                  << " diff " << diff 
                  << " diff2 " << diff2 
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


    float epsilon_translation = 1e-3 ; 
    float epsilon = 1e-5 ; 
    float diff = compDiff(isirit, i_trs );
    float diff2 = compDiff2(isirit, i_trs, false, epsilon, epsilon_translation );
    float diffFractional = compDiff2(isirit, i_trs, true, epsilon, epsilon_translation  );

    //float diffFractionalMax = 1e-4 ;  // <-- small row3 positions trip this  
    float diffFractionalMax = 1e-3 ; 

    bool match = diffFractional < diffFractionalMax ; 

    if(!match)
    {
       std::cout << "nglmext::invert_trs"
                 << " polar_decomposition inverse and straight inverse are mismatched "
                 << " epsilon " << epsilon
                 << " diff " << diff 
                 << " diff2 " << diff2 
                 << " diffFractional " << diffFractional
                 << " diffFractionalMax " << diffFractionalMax
                 << std::endl << gpresent("trs", trs)
                 << std::endl << gpresent("isirit", isirit)
                 << std::endl << gpresent("i_trs ",i_trs)
                 << std::endl ; 

    

        for(unsigned i=0 ; i < 4 ; i++)
        {
            for(unsigned j=0 ; j < 4 ; j++)
            {

                float a = isirit[i][j] ;
                float b = i_trs[i][j] ;

                float da = compDiff2(a,b, false, epsilon);
                float df = compDiff2(a,b, true , epsilon);

                bool ijmatch = df < diffFractionalMax ;

                std::cout << "[" 
                          << ( ijmatch ? "" : "**" ) 
                          << std::setw(10) << a
                          << ":"
                          << std::setw(10) << b
                          << ":"
                          << std::setw(10) << da
                          << ":"
                          << std::setw(10) << df
                          << ( ijmatch ? "" : "**" ) 
                          << "]"
                           ;
            }
            std::cout << std::endl; 
        }
    }
    assert(match);
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


/*
In [1]: a = 2.16489e-17

In [2]: b = 0 

In [3]: (a+b)/2
Out[3]: 1.082445e-17

In [4]: avg = (a+b)/2

In [5]: ab = a-b 

In [6]: ab/avg
Out[6]: 2.0

*/

float nglmext::compDiff2(const float a_ , const float b_, bool fractional, float epsilon  )
{
    float a = fabsf(a_) < epsilon  ? 0.f : a_ ; 
    float b = fabsf(b_) < epsilon  ? 0.f : b_ ; 

    float d = fabsf(a - b);
    if(fractional) d /= (a+b)/2.f ; 
    return d ; 
}

float nglmext::compDiff2(const glm::mat4& a_ , const glm::mat4& b_, bool fractional, float epsilon, float epsilon_translation )
{
    float a, b, d, maxdiff = 0.f ; 
    for(unsigned i=0 ; i < 4 ; i++){
    for(unsigned j=0 ; j < 4 ; j++)
    { 
        a = a_[i][j] ; 
        b = b_[i][j] ; 
        d = compDiff2(a, b, fractional, i == 3 ? epsilon_translation : epsilon );
        if( d > maxdiff ) maxdiff = d ; 
    }
    }
    return maxdiff ; 
}




glm::mat4 nglmext::make_translate(const float x, const float y, const float z)
{
   glm::vec3 tlate(x,y,z);
   return make_translate(tlate);
}
glm::mat4 nglmext::make_rotate(const float x, const float y, const float z, const float w)
{
   glm::vec4 aa(x,y,z,w);
   return make_rotate(aa);
}
glm::mat4 nglmext::make_scale(const float x, const float y, const float z)
{
   glm::vec3 tsc(x,y,z);
   return make_scale(tsc);
}




glm::mat4 nglmext::make_translate(const glm::vec3& tlat)
{
    return glm::translate(glm::mat4(1.f), tlat);
}



glm::mat4 nglmext::make_rotate(const glm::vec4& axis_angle)
{
    float angle = nglmext::angle_radians(axis_angle.w) ; 
    return glm::rotate(glm::mat4(1.f), angle , glm::vec3(axis_angle)) ;
}
glm::mat4 nglmext::make_scale(const glm::vec3& scal)
{
    return glm::scale(glm::mat4(1.f), scal);
}


float nglmext::angle_radians(float angle_degrees)
{
    return glm::pi<float>()*angle_degrees/180.f ; 
}

glm::mat4 nglmext::make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal )
{
    glm::mat4 mat(1.f) ;
    
    float angle = nglmext::angle_radians(axis_angle.w) ; 

    for(unsigned i=0 ; i < order.length() ; i++)
    {
        switch(order[i])
        {
           case 's': mat = glm::scale(mat, scal)         ; break ; 
           case 'r': mat = glm::rotate(mat, angle , glm::vec3(axis_angle)) ; break ; 
           case 't': mat = glm::translate(mat, tlat )    ; break ; 
        }
    }
    // for fourth column translation unmodified the "t" must come last, ie "trs" ?? shouldnt that be order="srt"
    //
    // Despite this intution from reading the code  there is the row-major business too..
    //
    // See tests/NGLMExt.cc:test_make_transform it shows that 
    // order argument "trs" somewhat confusingly : does scaling before translation, 
    // order argument "srt" somewhat confusingly : does translation before scaling

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


std::string nmat4triple::digest() const 
{
    return SDigest::digest( (void*)this, sizeof(nmat4triple) );
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
     t(t_),
     v(nglmext::invert_trs(t))
{
}


nmat4triple::nmat4triple(const float* data ) 
     : 
     t(glm::make_mat4(data)),
     v(nglmext::invert_trs(t)),
     q(glm::transpose(v))
{
}

nmat4triple::nmat4triple(const glm::mat4& t_ ) 
     : 
     t(t_),
     v(nglmext::invert_trs(t)),
     q(glm::transpose(v))
{
}

const nmat4triple* nmat4triple::clone() const 
{
    return new nmat4triple(t,v,q);
}



glm::vec3 nmat4triple::apply_transform_t(const glm::vec3& p_, const float w) const 
{
    ntransformer tr(t, w);
    return tr(p_); 
}

glm::vec3 nmat4triple::apply_transform_v(const glm::vec3& p_, const float w) const 
{
    ntransformer tr(v, w);
    return tr(p_); 
}

void nmat4triple::apply_transform_t(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w) const 
{
    ntransformer tr(t, w);
    std::transform(src.begin(), src.end(), std::back_inserter(dst), tr );
}
void nmat4triple::apply_transform_v(std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& src, float w) const 
{
    ntransformer tr(v, w);
    std::transform(src.begin(), src.end(), std::back_inserter(dst), tr );
}




const nmat4triple* nmat4triple::make_translate( const glm::vec3& tlate )
{
    glm::mat4 t = nglmext::make_translate(tlate);
    return new nmat4triple(t);
}
const nmat4triple* nmat4triple::make_scale( const glm::vec3& tsca )
{
    glm::mat4 s = nglmext::make_scale(tsca);
    return new nmat4triple(s);
}
const nmat4triple* nmat4triple::make_rotate( const glm::vec4& trot )
{
    glm::mat4 r = nglmext::make_rotate(trot);
    return new nmat4triple(r);
}


const nmat4triple* nmat4triple::make_translate( const float x, const float y, const float z)
{
    glm::vec3 tmp(x,y,z);
    return make_translate(tmp);
}
const nmat4triple* nmat4triple::make_scale( const float x, const float y, const float z)
{
    glm::vec3 tmp(x,y,z);
    return make_scale(tmp);
}
const nmat4triple* nmat4triple::make_rotate( const float x, const float y, const float z, const float w)
{
    glm::vec4 tmp(x,y,z,w);
    return make_rotate(tmp);
}





const nmat4triple* nmat4triple::product(const nmat4triple* a, const nmat4triple* b, bool reverse)
{
    std::vector<const nmat4triple*> triples ; 
    triples.push_back(a);
    triples.push_back(b);
    return nmat4triple::product( triples, reverse );
}

const nmat4triple* nmat4triple::product(const nmat4triple* a, const nmat4triple* b, const nmat4triple* c, bool reverse)
{
    std::vector<const nmat4triple*> triples ; 
    triples.push_back(a);
    triples.push_back(b);
    triples.push_back(c);
    return nmat4triple::product( triples, reverse );
}

const nmat4triple* nmat4triple::product(const std::vector<const nmat4triple*>& triples, bool reverse )
{
/*
    Use *reverse=true* when the triples are in reverse heirarchical order, ie when
    they have been collected by starting from the leaf node and then following parent 
    links back up to the root node. 
*/
    unsigned ntriples = triples.size();
    if(ntriples==0) return NULL ; 
    if(ntriples==1) return triples[0] ; 

    glm::mat4 t(1.0) ; 
    glm::mat4 v(1.0) ; 

    for(unsigned i=0,j=ntriples-1 ; i < ntriples ; i++,j-- )
    {
        // inclusive indices:
        //     i: 0 -> ntriples - 1      ascending 
        //     j: ntriples - 1 -> 0      descending (from last transform down to first)
        //
        const nmat4triple* ii = triples[reverse ? j : i] ;  // with reverse: start from the last (ie root node)
        const nmat4triple* jj = triples[reverse ? i : j] ;  // with reverse: start from the first (ie leaf node)

        t *= ii->t ;   
        v *= jj->v ;  // inverse-transform product in opposite order
    }

    // is this the appropriate transform and inverse transform multiplication order ?
    // ... triples order is from the leaf back to the root   

    glm::mat4 q = glm::transpose(v);
    return new nmat4triple(t, v, q) ; 
}


const nmat4triple* nmat4triple::make_identity()
{
    glm::mat4 identity(1.f); 
    return new nmat4triple(identity);
}


const nmat4triple* nmat4triple::make_translated(const glm::vec3& tlate, bool reverse, const char* user ) const 
{
    // reverse:true means the tlate happens at the root 
    // reverse:false means the tlate happens at the leaf
    return make_translated(this, tlate, reverse, user );
}

const nmat4triple* nmat4triple::make_translated(const nmat4triple* src, const glm::vec3& tlate, bool reverse, const char* user)
{ 
    glm::mat4 tra = glm::translate(glm::mat4(1.f), tlate);
    return make_transformed(src, tra, reverse, user );
}

const nmat4triple* nmat4triple::make_transformed(const nmat4triple* src, const glm::mat4& txf, bool reverse, const char* /*user*/)
{
    // reverse:true means the transform ordering is from leaf to root 
    // so when wishing to extend the hierarchy with a higher level root transform, 
    // that means just pushing another transform on the end of the existing vector

/*
    LOG(info) << "nmat4triple::make_transformed" 
              << " user " << user
              ;
*/

    nmat4triple perturb( txf );
    std::vector<const nmat4triple*> triples ; 

    // HMM its confusing to reverse here 
    // because reversal is also done in nmat4triple::product
    // so they will counteract ??
    // Who uses this ?
    
    // used by GMergedMesh::mergeSolidAnalytic/GParts::applyPlacementTransform
    //assert(0);

    if(reverse)
    { 
        triples.push_back(src);    
        triples.push_back(&perturb);
    }
    else
    {
        triples.push_back(&perturb);
        triples.push_back(src);    
    }

    const nmat4triple* transformed = nmat4triple::product( triples, reverse );  
    return transformed ; 
}


void nglmext::transform_planes( NPY<float>* plan_buffer, const glm::mat4& placement ) //static
{
    // Formerly all geometry that required planes (eg trapezoids) 
    // was part of instanced solids... so this was not needed.
    // BUT for debugging it is useful to be able to operate in global mode
    // whilst testing small subsets of geometry
    //   
    //if(nplane > 0 ) assert(0 && "plane placement not implemented" );
    assert(plan_buffer->hasShape(-1,4));
    unsigned num_plane = plan_buffer->getNumItems();
    for(unsigned i=0 ; i < num_plane ; i++ )
    {    
        glm::vec4 pl = plan_buffer->getQuad(i);
        nplane  plane = make_plane(pl);
        glm::vec4 tpl = plane.make_transformed(placement);

        plan_buffer->setQuad( tpl, i ); 
    }    
}


void nmat4triple::dump( const NPY<float>* buf, const char* msg)
{
    LOG(info) << msg ; 
    assert(buf->hasShape(-1,3,4,4));
    unsigned ni = buf->getNumItems();  
    for(unsigned i=0 ; i < ni ; i++)
    {
        nmat4triple* tvq = buf->getMat4TriplePtr(i) ;
        std::cout << std::setw(3) << i << " tvq " << *tvq << std::endl ;  
    }
}


void nmat4triple::dump( const float* data4x4, const char* msg )
{
    LOG(info) << msg ; 
    nmat4triple* tvq = new nmat4triple(data4x4)  ;
    std::cout << " tvq " << *tvq << std::endl ;  
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


std::ostream& operator<< (std::ostream& out, const nmat4triple& triple)
{
    out 
       << std::endl 
       << gpresent( "triple.t",  triple.t ) 
       << std::endl 
       << gpresent( "triple.v",  triple.v ) 
       << std::endl 
       << gpresent( "triple.q",  triple.q ) 
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








