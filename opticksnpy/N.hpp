#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NSDF.hpp"

struct nnode ; 
struct nmat4triple ;
struct NSceneConfig ; 
class NNodePoints ; 

/*
N : debugging structural nodes/SDF
======================================

N ctor collects eg pp.model CSG frame surface band points, 
and transforms them with the ctor argument placement transform 
into pp.local points, also the N ctor instanciates
an NSDF instance with the sdf obtained from the nnode argument 
and the inverse of the transform ... 

Subsequenently calls to N::classify with points 
in the operation frame uses the inverse of the placement
transform to allow appropriate SDF values to be obtained.

Using identity transform for parent node and n->transform
for this node ... allows cross frame comparisons. 
The p->transform is not relevant as are treating the parent
as the base frame.

It may be less confusing to  think about being in the parent 
node with a placed child.

Recall that surface points are obtained from parametric primitives
by CSG/model frame SDF comparisons ... so surface points are in a narrow band around SDF zero
depending on the surface_epsilon used to collect them from the primitives.

*/
 
struct NPY_API N 
{
    N(nnode* node, const nmat4triple* transform, const NSceneConfig* config=NULL, float surface_epsilon=1e-6f );
    glm::uvec4 classify(const std::vector<glm::vec3>& qq, float epsilon, unsigned expect, bool dump=false);
    void dump_points(const char* msg);
    std::string desc() const ;

    const nnode*           node ;
    const nmat4triple*     transform ; 
    const NSceneConfig*    config ; 
    NNodePoints*           points ; 

    NSDF                   nsdf ; 
    glm::uvec4             tots ;
    unsigned               num ; 
 
    std::vector<glm::vec3> model ; 
    std::vector<glm::vec3> local ; 

};



