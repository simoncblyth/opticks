#pragma once

#include <string>
#include <vector>
#include <functional>
#include <glm/fwd.hpp>

#include "OpticksCSG.h"
#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"

struct nbbox ; 
struct npart ; 

// NGLMExt
struct nmat4pair ; 
struct nmat4triple ; 

struct NPY_API nnode 
{
    virtual float operator()(float px, float py, float pz) ;

    virtual void dump(const char* msg="nnode::dump");
    virtual const char* csgname(); 
    virtual nbbox bbox();
    virtual npart part();
    virtual unsigned maxdepth();
    virtual unsigned _maxdepth(unsigned depth);
    virtual std::string desc();

    static void Tests(std::vector<nnode*>& nodes );
    static void Init(nnode& n, OpticksCSG_t type, nnode* left=NULL, nnode* right=NULL);

    std::function<float(float,float,float)> sdf();


    void update_gtransforms();
    static void update_gtransforms_r(nnode* node);

    nmat4triple* global_transform(); 
    static nmat4triple* global_transform(nnode* n); 

    void collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs);

    void dump_prim(const char* msg="dump_prim", int verbosity=1 ) ;
    void collect_prim(std::vector<nnode*>& prim) ;
    static void collect_prim_r(std::vector<nnode*>& prim, nnode* node) ;


    unsigned idx ; 
    OpticksCSG_t type ;  
    nnode* left ; 
    nnode* right ; 
    nnode* parent ; 
    const char* label ; 

    nmat4triple* transform ; 
    nmat4triple* gtransform ; 
    unsigned   gtransform_idx ; 

    nquad param ; 
    nquad param1 ; 
    nquad param2 ; 
    nquad param3 ; 
};

// hmm perhaps easier to switch on these ?? instead
// of having separate types ? 

struct NPY_API nunion : nnode {
    float operator()(float x, float y, float z) ;
};
struct NPY_API nintersection : nnode {
    float operator()(float x, float y, float z);
};
struct NPY_API ndifference : nnode {
    float operator()(float x, float y, float z);
};


inline NPY_API nunion make_union(nnode* left, nnode* right)
{
    nunion n ;         nnode::Init(n, CSG_UNION , left, right ); return n ; 
}
inline NPY_API nintersection make_intersection(nnode* left, nnode* right)
{
    nintersection n ;  nnode::Init(n, CSG_INTERSECTION , left, right ); return n ;
}
inline NPY_API ndifference make_difference(nnode* left, nnode* right)
{
    ndifference n ;    nnode::Init(n, CSG_DIFFERENCE , left, right ); return n ;
}



