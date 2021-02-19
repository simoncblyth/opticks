#include "NGLMExt.hpp"
#include "nmat4triple_.hpp"
#include "GLMFormat.hpp"
#include "SDigest.hh"
#include "PLOG.hh"


template<typename T>
const plog::Severity nmat4triple_<T>::LEVEL = PLOG::EnvLevel("nmat4triple_", "DEBUG"); 


template<typename T>
nmat4triple_<T>::nmat4triple_( const glm::tmat4x4<T>& transform, const glm::tmat4x4<T>& inverse, const glm::tmat4x4<T>& inverse_T ) 
    : 
    match(true),
    t(transform), 
    v(inverse), 
    q(inverse_T) 
{
} 

template<typename T>
nmat4triple_<T>::nmat4triple_(const T* data ) 
    : 
    match(true),
    t(glm::make_mat4x4(data)),
    v(nglmext::invert_trs(t, match)),
    q(glm::transpose(v))
{
    if(!match) LOG(error) << " mis-match " ; 
}

template<typename T>
nmat4triple_<T>::nmat4triple_(const glm::tmat4x4<T>& t_ ) 
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

template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::clone() const 
{
    return new nmat4triple_<T>(t,v,q);
}



template<typename T>
glm::tvec3<T> nmat4triple_<T>::apply_transform_t(const glm::tvec3<T>& p_, const T w) const 
{
    ntransformer tr(t, w);
    return tr(p_); 
}

template<typename T>
glm::tvec3<T> nmat4triple_<T>::apply_transform_v(const glm::tvec3<T>& p_, const T w) const 
{
    ntransformer tr(v, w);
    return tr(p_); 
}

template<typename T>
void nmat4triple_<T>::apply_transform_t(std::vector<glm::tvec3<T>>& dst, const std::vector<glm::tvec3<T>>& src, T w) const 
{
    ntransformer tr(t, w);
    std::transform(src.begin(), src.end(), std::back_inserter(dst), tr );
}
template<typename T>
void nmat4triple_<T>::apply_transform_v(std::vector<glm::tvec3<T>>& dst, const std::vector<glm::tvec3<T>>& src, T w) const 
{
    ntransformer tr(v, w);
    std::transform(src.begin(), src.end(), std::back_inserter(dst), tr );
}

template<typename T>
bool nmat4triple_<T>::is_equal_to(const nmat4triple_<T>* other, T eps) const 
{
    assert( other ) ; 
    T dt = nglmext::compDiff_<T>(t, other->t);
    T dv = nglmext::compDiff_<T>(v, other->v);
    T dq = nglmext::compDiff_<T>(q, other->q);
    return dt < eps && dv < eps && dq < eps ; 
}




template<typename T>
glm::tvec3<T> nmat4triple_<T>::get_translation() const 
{
    glm::tvec3<T> tla(t[3]) ;  
    return tla ; 
}

template<typename T>
bool nmat4triple_<T>::is_translation_only(T eps) const 
{
    const glm::tmat3x3<T> i3(T(1)); 
    const glm::tmat3x3<T> t3(t) ; 
    T dt = nglmext::compDiff_<T>(t3, i3);
    return dt < eps ; 
}

template<typename T>
bool nmat4triple_<T>::is_identity(T eps) const 
{
    glm::tmat4x4<T> id(T(1)) ; 
    T dt = nglmext::compDiff_<T>(t, id);
    T dv = nglmext::compDiff_<T>(v, id);
    T dq = nglmext::compDiff_<T>(q, id);
    return dt < eps && dv < eps && dq < eps ; 
}



template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_transform( 
           const T x0, const T y0, const T z0, const T w0,
           const T x1, const T y1, const T z1, const T w1, 
           const T x2, const T y2, const T z2, const T w2, 
           const T x3, const T y3, const T z3, const T w3 
       )  // static
{
    glm::tmat4x4<T> t(T(1));

    t[0] = glm::tvec4<T>(x0,y0,z0,w0); 
    t[1] = glm::tvec4<T>(x1,y1,z1,w1); 
    t[2] = glm::tvec4<T>(x2,y2,z2,w2); 
    t[3] = glm::tvec4<T>(x3,y3,z3,w3); 

    return new nmat4triple_<T>(t);
}


template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_translate( const glm::tvec3<T>& tlate )
{
    glm::tmat4x4<T> t = nglmext::make_translate(tlate);
    return new nmat4triple_<T>(t);
}
template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_scale( const glm::tvec3<T>& tsca )
{
    glm::tmat4x4<T> s = nglmext::make_scale(tsca);
    return new nmat4triple_<T>(s);
}
template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_rotate( const glm::tvec4<T>& trot )
{
    glm::tmat4x4<T> r = nglmext::make_rotate(trot);
    return new nmat4triple_<T>(r);
}


template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_translate( const T x, const T y, const T z)
{
    glm::tvec3<T> tmp(x,y,z);
    return make_translate(tmp);
}
template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_scale( const T x, const T y, const T z)
{
    glm::tvec3<T> tmp(x,y,z);
    return make_scale(tmp);
}
template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_rotate( const T x, const T y, const T z, const T w)
{
    glm::tvec4<T> tmp(x,y,z,w);
    return make_rotate(tmp);
}





template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::product(const nmat4triple_<T>* a, const nmat4triple_<T>* b, bool reverse)
{
    std::vector<const nmat4triple_<T>*> triples ; 
    triples.push_back(a);
    triples.push_back(b);
    return nmat4triple_<T>::product( triples, reverse );
}

template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::product(const nmat4triple_<T>* a, const nmat4triple_<T>* b, const nmat4triple_<T>* c, bool reverse)
{
    std::vector<const nmat4triple_<T>*> triples ; 
    triples.push_back(a);
    triples.push_back(b);
    triples.push_back(c);
    return nmat4triple_<T>::product( triples, reverse );
}

template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::product(const std::vector<const nmat4triple_<T>*>& triples, bool reverse )
{
/*
    Use *reverse=true* when the triples are in reverse heirarchical order, ie when
    they have been collected by starting from the leaf node and then following parent 
    links back up to the root node. 
*/
    unsigned ntriples = triples.size();
    if(ntriples==0) return NULL ; 
    if(ntriples==1) return triples[0] ; 

    glm::tmat4x4<T> t(T(1)) ; 
    glm::tmat4x4<T> v(T(1)) ; 

    for(unsigned i=0,j=ntriples-1 ; i < ntriples ; i++,j-- )
    {
        // inclusive indices:
        //     i: 0 -> ntriples - 1      ascending 
        //     j: ntriples - 1 -> 0      descending (from last transform down to first)
        //
        const nmat4triple_<T>* ii = triples[reverse ? j : i] ;  // with reverse: start from the last (ie root node)
        const nmat4triple_<T>* jj = triples[reverse ? i : j] ;  // with reverse: start from the first (ie leaf node)

        t *= ii->t ;   
        v *= jj->v ;  // inverse-transform product in opposite order
    }

    // is this the appropriate transform and inverse transform multiplication order ?
    // ... triples order is from the leaf back to the root   

    glm::tmat4x4<T> q = glm::transpose(v);
    return new nmat4triple_<T>(t, v, q) ; 
}


template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_identity()
{
    glm::tmat4x4<T> identity(T(1)); 
    return new nmat4triple_<T>(identity);
}

/**
nmat4triple_<T>::make_translated
------------------------------

reverse:true 
    means the tlate happens at the root 

reverse:false 
    means the tlate happens at the leaf

**/

template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_translated(const glm::tvec3<T>& tlate, bool reverse, const char* user, bool& match ) const 
{
    return make_translated(this, tlate, reverse, user, match );
}

template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_translated(const nmat4triple_<T>* src, const glm::tvec3<T>& tlate, bool reverse, const char* user, bool& match)
{ 
    glm::tmat4x4<T> tra = glm::translate(glm::tmat4x4<T>(T(1)), tlate);
    return make_transformed(src, tra, reverse, user, match );
}

/**
nmat4triple_<T>::make_transformed
-------------------------------

reverse=true
     means the transform ordering is from leaf to root 
     so when wishing to extend the hierarchy with a higher level root transform, 
     that means just pushing another transform on the end of the existing vector

// HMM its confusing to reverse here 
// because reversal is also done in nmat4triple_<T>::product
// so they will counteract ??
// Who uses this ?

// used by GMergedMesh::mergeSolidAnalytic/GParts::applyPlacementTransform

**/

template<typename T>
const nmat4triple_<T>* nmat4triple_<T>::make_transformed(const nmat4triple_<T>* src, const glm::tmat4x4<T>& txf, bool reverse, const char* user, bool& match) // static
{
    LOG(LEVEL) << "[ " << user ; 

    nmat4triple_<T> perturb( txf );
    if(perturb.match == false)
    {
        LOG(error) << "perturb.match false : precision issue in inverse ? " ; 
    }

    match = perturb.match ; 

    std::vector<const nmat4triple_<T>*> triples ; 

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

    const nmat4triple_<T>* transformed = nmat4triple_<T>::product( triples, reverse );  

    LOG(LEVEL) << "] " << user ; 
    return transformed ; 
}



template<typename T>
void nmat4triple_<T>::dump( const NPY<T>* buf, const char* msg)
{
    LOG(info) << msg ; 
    assert(buf->hasShape(-1,3,4,4));
    unsigned ni = buf->getNumItems();  
    for(unsigned i=0 ; i < ni ; i++)
    {
        //nmat4triple_<T>* tvq = buf->getMat4TriplePtr(i) ;

        glm::tmat4x4<T> t = buf->getMat4_(i, 0);
        glm::tmat4x4<T> v = buf->getMat4_(i, 1);
        glm::tmat4x4<T> q = buf->getMat4_(i, 2);

        nmat4triple_<T>* tvq = new nmat4triple_<T>(t, v, q) ; 
        std::cout << std::setw(3) << i << " tvq " << *tvq << std::endl ;  
    }
}


template<typename T>
void nmat4triple_<T>::dump( const T* data4x4, const char* msg )
{
    LOG(info) << msg ; 
    nmat4triple_<T>* tvq = new nmat4triple_<T>(data4x4)  ;
    std::cout << " tvq " << *tvq << std::endl ;  
}


template<typename T>
std::string nmat4triple_<T>::digest() const 
{
    return SDigest::digest( (void*)this, sizeof(nmat4triple_<T>) );
}




template<typename T>
std::ostream& operator<< (std::ostream& out, const nmat4triple_<T>& triple)
{
    out 
       << std::endl 
       << gpresent__( "triple.t",  triple.t ) 
       << std::endl 
       << gpresent__( "triple.v",  triple.v ) 
       << std::endl 
       << gpresent__( "triple.q",  triple.q ) 
       << std::endl 
       ; 

    return out;
}


template struct nmat4triple_<float> ;  
template struct nmat4triple_<double> ;  


