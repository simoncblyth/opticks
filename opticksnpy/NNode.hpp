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


/*
typedef enum
{  
   FRAME_MODEL, 
   FRAME_LOCAL, 
   FRAME_GLOBAL 

} NNodeFrameType ;


    static const char* FRAME_MODEL_ ;
    static const char* FRAME_LOCAL_;
    static const char* FRAME_GLOBAL_ ;

    static const char* FrameType(NNodeFrameType fr);

*/

#include "NNodeEnum.hpp"

struct NPY_API nnode 
{
    virtual float operator()(float px, float py, float pz) const  ;
    virtual float sdf_(const glm::vec3& pos, NNodeFrameType fr) const ;

    static nnode* load(const char* treedir, int verbosity);
    static void AdjustToFit(nnode* node, const nbbox& bb, float scale) ;

    virtual void dump(const char* msg="nnode::dump") const ;
    virtual void pdump(const char* msg="nnode::pdump") const ; 
    virtual const char* csgname(); 
    virtual nbbox bbox() const ;

    void composite_bbox( nbbox& bb ) const ;

    virtual npart part();
    virtual unsigned maxdepth();
    virtual unsigned _maxdepth(unsigned depth);
    virtual std::string desc() const ;
    std::string tag() const ;

    static void Tests(std::vector<nnode*>& nodes );
    static void Init(nnode& n, OpticksCSG_t type, nnode* left=NULL, nnode* right=NULL);

    unsigned uncoincide();
    //bool can_uncoincide(const nnode* a, const nnode* b) const ;

    void dumpSurfacePointsAll(const char* msg, NNodeFrameType fr) const ;
    void getSurfacePointsAll(       std::vector<glm::vec3>& surf,        unsigned level, int margin, NNodeFrameType fr) const ;
    void getSurfacePoints(          std::vector<glm::vec3>& surf, int s, unsigned level, int margin, NNodeFrameType fr) const ;
    void getCoincidentSurfacePoints(std::vector<nuv>& coincident, int s, unsigned level, int margin, const nnode* other, float epsilon, NNodeFrameType fr) const ;
    void getCoincident(             std::vector<nuv>& coincident, const nnode* other, float epsilon=1e-5f, unsigned level=1, int margin=1, NNodeFrameType fr=FRAME_LOCAL) const ;


    std::function<float(float,float,float)> sdf() const ;

    glm::vec3 gseeddir();  // override if needed

    virtual unsigned  par_nsurf() const ;
    virtual glm::vec3 par_pos(const nuv& uv) const ;
    virtual glm::vec3 par_pos_(const nuv& uv, NNodeFrameType fr) const ;
    virtual int       par_euler() const ; 
    virtual unsigned  par_nvertices(unsigned nu, unsigned nv) const ;
    virtual void      nudge(unsigned s, float delta);


    void update_gtransforms();
    static void update_gtransforms_r(nnode* node);

    nmat4triple* global_transform(); 
    static nmat4triple* global_transform(nnode* n); 

    glm::vec3 apply_gtransform(const glm::vec4& v_) const ;

    void collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs, int verbosity=0);

    void dump_full(const char* msg="nnode::dump_full") const ;
    void dump_transform(const char* msg="nnode::dump_transform") const ;
    void dump_gtransform(const char* msg="nnode::dump_gtransform") const ;
    void dump_prim(const char* msg="nnode::dump_prim") const ;
    void collect_prim(std::vector<const nnode*>& prim) const ;
    static void collect_prim_r(std::vector<const nnode*>& prim, const nnode* node) ;

    bool is_primitive() const ;
    bool is_bileaf() const ;
    bool has_planes();
    unsigned planeIdx();
    unsigned planeNum();

    unsigned idx ; 
    OpticksCSG_t type ;  
    nnode* left ; 
    nnode* right ; 
    nnode* parent ; 
    nnode* other ; 
    const char* label ; 

    nmat4triple* transform ; 
    nmat4triple* gtransform ; 
    unsigned   gtransform_idx ; 
    bool  complement ; 
    int verbosity ; 

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



