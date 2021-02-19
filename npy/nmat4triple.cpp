#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include "GLMFormat.hpp"
#include "SDigest.hh"
#include "PLOG.hh"


const plog::Severity nmat4triple::LEVEL = PLOG::EnvLevel("nmat4triple", "DEBUG"); 


nmat4triple::nmat4triple( const glm::mat4& transform, const glm::mat4& inverse, const glm::mat4& inverse_T ) 
    : 
    match(true),
    t(transform), 
    v(inverse), 
    q(inverse_T) 
{
} 

nmat4triple::nmat4triple(const float* data ) 
    : 
    match(true),
    t(glm::make_mat4(data)),
    v(nglmext::invert_trs(t, match)),
    q(glm::transpose(v))
{
    if(!match) LOG(error) << " mis-match " ; 
}

nmat4triple::nmat4triple(const glm::mat4& t_ ) 
    : 
    match(true),
    t(t_),
    v(nglmext::invert_trs(t, match)),
    q(glm::transpose(v))
{
    if(!match)
    {
        LOG(error) << " mis-match " ; 
        //std::raise(SIGINT); 
    }
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

bool nmat4triple::is_equal_to(const nmat4triple* other, float eps) const 
{
    assert( other ) ; 
    float dt = nglmext::compDiff(t, other->t);
    float dv = nglmext::compDiff(v, other->v);
    float dq = nglmext::compDiff(q, other->q);
    return dt < eps && dv < eps && dq < eps ; 
}




glm::vec3 nmat4triple::get_translation() const 
{
    glm::vec3 tla(t[3]) ;  
    return tla ; 
}

bool nmat4triple::is_translation_only(float eps) const 
{
    const glm::mat3 i3(1.f); 
    const glm::mat3 t3(t) ; 
    float dt = nglmext::compDiff(t3, i3);
    return dt < eps ; 
}

bool nmat4triple::is_identity(float eps) const 
{
    glm::mat4 id(1.0) ; 
    float dt = nglmext::compDiff(t, id);
    float dv = nglmext::compDiff(v, id);
    float dq = nglmext::compDiff(q, id);
    return dt < eps && dv < eps && dq < eps ; 
}



const nmat4triple* nmat4triple::make_transform( 
           const float x0, const float y0, const float z0, const float w0,
           const float x1, const float y1, const float z1, const float w1, 
           const float x2, const float y2, const float z2, const float w2, 
           const float x3, const float y3, const float z3, const float w3 
       )  // static
{
    glm::mat4 t(1);

    t[0] = glm::vec4(x0,y0,z0,w0); 
    t[1] = glm::vec4(x1,y1,z1,w1); 
    t[2] = glm::vec4(x2,y2,z2,w2); 
    t[3] = glm::vec4(x3,y3,z3,w3); 

    return new nmat4triple(t);
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

/**
nmat4triple::make_translated
------------------------------

reverse:true 
    means the tlate happens at the root 

reverse:false 
    means the tlate happens at the leaf

**/

const nmat4triple* nmat4triple::make_translated(const glm::vec3& tlate, bool reverse, const char* user, bool& match ) const 
{
    return make_translated(this, tlate, reverse, user, match );
}

const nmat4triple* nmat4triple::make_translated(const nmat4triple* src, const glm::vec3& tlate, bool reverse, const char* user, bool& match)
{ 
    glm::mat4 tra = glm::translate(glm::mat4(1.f), tlate);
    return make_transformed(src, tra, reverse, user, match );
}

/**
nmat4triple::make_transformed
-------------------------------

reverse=true
     means the transform ordering is from leaf to root 
     so when wishing to extend the hierarchy with a higher level root transform, 
     that means just pushing another transform on the end of the existing vector

// HMM its confusing to reverse here 
// because reversal is also done in nmat4triple::product
// so they will counteract ??
// Who uses this ?

// used by GMergedMesh::mergeSolidAnalytic/GParts::applyPlacementTransform

**/

const nmat4triple* nmat4triple::make_transformed(const nmat4triple* src, const glm::mat4& txf, bool reverse, const char* user, bool& match) // static
{
    LOG(LEVEL) << "[ " << user ; 

    nmat4triple perturb( txf );
    if(perturb.match == false)
    {
        LOG(error) << "perturb.match false : precision issue in inverse ? " ; 
    }

    match = perturb.match ; 

    std::vector<const nmat4triple*> triples ; 

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

    LOG(LEVEL) << "] " << user ; 
    return transformed ; 
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


std::string nmat4triple::digest() const 
{
    return SDigest::digest( (void*)this, sizeof(nmat4triple) );
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







