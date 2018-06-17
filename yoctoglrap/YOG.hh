#pragma once

#include <string>
#include <vector>

#include "YOG_API_EXPORT.hh"
template <typename T> class NPY ; 

#include <glm/fwd.hpp>

/**
YOG : intermediate geometry tree
===================================

The YOG geometry model (Sc, Nd, Mh) is used as an intermediary model 
at a slightly higher level than ygltf, above the gltf details.

* YOG model holds the nodes and meshes and their association 
* typically there are many more nodes than meshes 
* YOG has no ygltf dependency 

The Geant4 volume tree is converted into YOG by calling YOG::Nd::add_node
from within the recursive X4PhysicalVolume::TraverseVolumeTree.
Subsequently Sc can be converted to ygltf and saved to file 
using YOG::TF.

The YOG model follows that provided in ../analytic/sc.py
which was used to convert a parsed GDML into non-renderable
GLTF for consumption by NGLTF/NScene.

* NB there is no public or private dependency of 
  YOG model on YoctoGL, it sits atop NPY : so it should live there perhaps ?

  * but NPY has gotten far to big, need to split that up : before 
    deciding where to put YOG 
  * the GMesh is held just as a pointer, despite no use of GGeo 
  * YoctoGLRap : SysRap NPY YoctoGL  



**/

struct nnode ; 
class GMesh ; 

namespace YOG {

struct Nd ; 
struct Mh ; 
struct Mt ; 

struct YOG_API Sc 
{
    Sc(int root_=0);

    int  root ; 
    std::vector<Nd*>  nodes ; 
    std::vector<Mh*>  meshes ;  // solids
    std::vector<Mt*>  materials ; 

    std::string desc() const ;

    int add_node(int   lvIdx, 
                 int   mtIdx,
                 const std::string& lvName, 
                 const std::string& pvName, 
                 const std::string& soName, 
                 const glm::mat4* transform, 
                 const std::string& boundary,
                 int   depth, 
                 bool  selected);  

    int add_test_node(int lvIdx);

    Nd* get_node(int nodeIdx) const ; 
    Mh* get_mesh_for_node(int nodeIdx) const ; 

    int lv2so(int lvIdx) const ;
    bool has_mesh(int lvIdx) const ; 

    int add_mesh(int   lvIdx,
                 int   mtIdx,
                 const std::string& lvName, 
                 const std::string& soName);

    int add_material(const std::string& matName); 
    int get_material_idx( const std::string& matName) const ;  // -1 if not found


};

struct YOG_API Nd
{
    int              ndIdx ; 
    int              soIdx ;   // mesh 
    const glm::mat4* transform ; 
    std::string      boundary ; 
    std::string      name ;   // pvname 
    int              depth ;
    const Sc*        scene ;   // TODO: maybe remove 
    bool             selected ; 
    int              parent ; 

    std::vector<int> children ; 
    std::string desc() const ;
};

struct YOG_API Mh
{
    int          lvIdx ; 
    int          mtIdx ;   
    std::string lvName ; 
    std::string soName ;
    int          soIdx ; 

    nnode*         csg ; 
    GMesh*        mesh ; 
    NPY<float>*    vtx ; 
    NPY<unsigned>* idx ; 

    std::string desc() const ;
};

struct YOG_API Mt
{
    std::string name ; 
    std::string desc() const ;
};





}  // namespace

