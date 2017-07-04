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
class NNodeDump ; 

// NGLMExt
struct nmat4pair ; 
struct nmat4triple ; 

#include "NNodeEnum.hpp"

struct NPY_API nnode 
{
    //virtual float operator()(const glm::vec3& p) const ;
    virtual float operator()(float px, float py, float pz) const  ;
    virtual float sdf_(const glm::vec3& pos, NNodeFrameType fr) const ;

    std::function<float(float,float,float)> sdf() const ;

    static nnode* load(const char* treedir, int verbosity);
    static void AdjustToFit(nnode* node, const nbbox& bb, float scale) ;

    virtual const char* csgname(); 
    virtual nbbox bbox() const ;

    void get_composite_bbox( nbbox& bb ) const ;
    void get_primitive_bbox( nbbox& bb ) const ;

    virtual npart part() const ;
    virtual unsigned maxdepth();
    virtual unsigned _maxdepth(unsigned depth);
    virtual std::string desc() const ;
    std::string tag() const ;

    static void Tests(std::vector<nnode*>& nodes );
    static void Init(nnode& n, OpticksCSG_t type, nnode* left=NULL, nnode* right=NULL);

    unsigned uncoincide(unsigned verbosity);
    //bool can_uncoincide(const nnode* a, const nnode* b) const ;


    glm::uvec4 getCompositePoints( std::vector<glm::vec3>& surf, unsigned level, int margin , unsigned pointmask, float epsilon, const glm::mat4* tr ) const ;
    glm::uvec4 selectBySDF(std::vector<glm::vec3>& dest, const std::vector<glm::vec3>& source, unsigned pointmask, float epsilon, const glm::mat4* tr) const ;

    void dumpPointsSDF(const std::vector<glm::vec3>& points, float epsilon ) const ;
    void dumpSurfacePointsAll(const char* msg, NNodeFrameType fr) const ;
    void getSurfacePointsAll(       std::vector<glm::vec3>& surf,        unsigned level, int margin, NNodeFrameType fr) const ;
    void getSurfacePoints(          std::vector<glm::vec3>& surf, int s, unsigned level, int margin, NNodeFrameType fr) const ;
    void getCoincidentSurfacePoints(std::vector<nuv>& coincident, int s, unsigned level, int margin, const nnode* other, float epsilon, NNodeFrameType fr) const ;
    void getCoincident(             std::vector<nuv>& coincident, const nnode* other, float epsilon=1e-5f, unsigned level=1, int margin=1, NNodeFrameType fr=FRAME_LOCAL) const ;



    glm::vec3 gseeddir() const ;  // override if needed

    glm::vec3 par_pos_(const nuv& uv, NNodeFrameType fr) const ;
    glm::vec3 par_pos_(const nuv& uv, const nmat4triple* triple) const ;
    glm::vec3 par_pos_local(const nuv& uv) const ;  // "transform"  local node frame
    glm::vec3 par_pos_global(const nuv& uv) const ; // "gtransform" CSG tree root node frame 


    virtual glm::vec3 par_pos_model(const nuv& uv) const ;
    virtual unsigned  par_nsurf() const ;
    virtual int       par_euler() const ; 
    virtual unsigned  par_nvertices(unsigned nu, unsigned nv) const ;
    virtual void      nudge(unsigned s, float delta);

    static void _par_pos_endcap(glm::vec3& pos,  const nuv& uv, const float r_, const float z_ ) ; 


    // see NNodeUncoincide
    virtual void increase_z2(float dz);
    virtual void decrease_z1(float dz);
    virtual float z1() const ; 
    virtual float z2() const ; 
    virtual float r1() const ; 
    virtual float r2() const ; 


    void update_gtransforms();
    static void update_gtransforms_r(nnode* node);

    const nmat4triple* global_transform(); 
    static const nmat4triple* global_transform(nnode* n); 

    glm::vec3 apply_gtransform(const glm::vec4& v_) const ;

    void collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs, int verbosity=0);



    virtual void dump(const char* msg=NULL) const ;
    virtual void pdump(const char* msg="nnode::pdump") const ; 

    void dump_label(const char* pfx, const char* msg=NULL) const ;
    void dump_full(const char* msg=NULL) const ;
    void dump_bbox(const char* msg=NULL ) const ;
    void dump_transform(const char* msg=NULL) const ;
    void dump_gtransform(const char* msg=NULL) const ;
    void dump_prim(const char* msg=NULL) const ;
    void dump_planes(const char* msg=NULL) const ;
  


    unsigned get_num_prim() const ;
    void collect_prim(std::vector<const nnode*>& prim) const ;
    static void collect_prim_r(std::vector<const nnode*>& prim, const nnode* node) ;

    void collect_prim_for_edit(std::vector<nnode*>& prim) ;
    static void collect_prim_for_edit_r(std::vector<nnode*>& prim, nnode* node) ;


    std::string get_type_mask_string() const ;
    unsigned    get_type_mask() const ;
    static void get_type_mask_r(const nnode* node, unsigned& tymsk);

    void set_treedir(const char* treedir) ; 

    bool is_znudge_capable() const ;
    bool is_operator() const ;
    bool is_primitive() const ;
    bool is_root() const ;
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
    const char* treedir ; 

    const nmat4triple* transform ; 
    const nmat4triple* gtransform ; 
    unsigned   gtransform_idx ; 
    bool  complement ; 
    int verbosity ; 

    nquad param ; 
    nquad param1 ; 
    nquad param2 ; 
    nquad param3 ; 

    std::vector<glm::vec4> planes ; 

    NNodeDump* _dump ;

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



