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
struct nuv ; 

// NGLMExt
struct nmat4pair ; 
struct nmat4triple ; 

//template <typename T> struct NOpenMesh ;


struct NPY_API nnode 
{
    virtual float operator()(float px, float py, float pz) const  ;

    static nnode* load(const char* treedir, unsigned verbosity);
    static void Scan( const nnode& node, const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& tt );

    virtual void dump(const char* msg="nnode::dump");
    virtual const char* csgname(); 
    virtual nbbox bbox() const ;
    virtual npart part();
    virtual unsigned maxdepth();
    virtual unsigned _maxdepth(unsigned depth);
    virtual std::string desc();

    static void Tests(std::vector<nnode*>& nodes );
    static void Init(nnode& n, OpticksCSG_t type, nnode* left=NULL, nnode* right=NULL);

    std::function<float(float,float,float)> sdf() const ;

    glm::vec3 gseeddir();  // override if needed

    virtual unsigned  par_nsurf() const ;
    virtual glm::vec3 par_pos(const nuv& uv) const ;
    virtual int       par_euler() const ; 
    virtual unsigned  par_nvertices(unsigned nu, unsigned nv) const ;

    void update_gtransforms();
    static void update_gtransforms_r(nnode* node);

    nmat4triple* global_transform(); 
    static nmat4triple* global_transform(nnode* n); 
    glm::vec3 apply_gtransform(const glm::vec4& v_);

    void collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs, int verbosity=0);

    void dump_prim(const char* msg="dump_prim", int verbosity=1 ) ;
    void collect_prim(std::vector<nnode*>& prim) ;
    static void collect_prim_r(std::vector<nnode*>& prim, nnode* node) ;

    bool has_planes();
    unsigned planeIdx();
    unsigned planeNum();

    unsigned idx ; 
    OpticksCSG_t type ;  
    nnode* left ; 
    nnode* right ; 
    nnode* parent ; 
    const char* label ; 

    nmat4triple* transform ; 
    nmat4triple* gtransform ; 
    unsigned   gtransform_idx ; 
    bool  complement ; 

    nquad param ; 
    nquad param1 ; 
    nquad param2 ; 
    nquad param3 ; 

    std::vector<nvec4> planes ; 

};

// hmm perhaps easier to switch on these ?? instead
// of having separate types ? 

struct NPY_API nunion : nnode {
    float operator()(float x, float y, float z) const ;
};
struct NPY_API nintersection : nnode {
    float operator()(float x, float y, float z) const ;
};
struct NPY_API ndifference : nnode {
    float operator()(float x, float y, float z) const ;
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



