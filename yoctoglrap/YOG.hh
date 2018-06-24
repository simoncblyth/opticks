#pragma once

#include <string>
#include <vector>

#include "YOG_API_EXPORT.hh"
template <typename T> class NPY ; 

struct nmat4triple ; 

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
class NCSG ; 
struct nmat4triple ; 
class GMesh ; 

namespace YOG {

struct Nd ; 
struct Mh ; 
struct Mt ; 

struct YOG_API Sc  // scene
{
    Sc(int root_=0);

    int  root ; 
    std::vector<Nd*>  nodes ; 
    std::vector<Mh*>  meshes ;  // solids
    std::vector<Mt*>  materials ; 

    std::string desc() const ;

    int get_num_meshes() const ;  
    int get_num_nodes() const ;  

    int add_node(int   lvIdx, 
                 int   mtIdx,
                 const std::string& lvName, 
                 const std::string& pvName, 
                 const std::string& soName, 
                 const nmat4triple* transform, 
                 const std::string& boundary,
                 int   depth, 
                 bool  selected,
                 Nd*   parent 
                 );  

    int add_test_node(int lvIdx);

    Nd* get_node(int nodeIdx) const ; 
    Mh* get_mesh_for_node(int nodeIdx) const ;  // node->mesh association via nd->soIdx

    int lv2so(int lvIdx) const ;
    bool has_mesh(int lvIdx) const ; 

    int add_mesh(int   lvIdx,    
                 int   mtIdx,
                 const std::string& lvName, 
                 const std::string& soName);  // mesh identity from lvIdx

    int add_material(const std::string& matName); 
    int get_material_idx( const std::string& matName) const ;  // -1 if not found


};

struct YOG_API Nd  // node
{
    int              ndIdx ; 
    int              soIdx ;   // mesh 
    const nmat4triple* transform ; 

    std::string      boundary ; 
    std::string      name ;   // pvname 
    int              depth ;
    const Sc*        scene ;   // TODO: maybe remove 
    bool             selected ; 
    Nd*              parent ; 
    Mh*              mh ; 

    std::vector<int> children ; 
    std::string desc() const ;
};

struct YOG_API Mh  // mesh
{
    int          lvIdx ; 
    int          mtIdx ;   
    std::string lvName ; 
    std::string soName ;
    int          soIdx ; 

    //const nnode*         csgnode ; 
    nnode*         csgnode ;     // not-const due to nudger
    const NCSG*          csg ; 
    const GMesh*         mesh ; 
    const NPY<float>*    vtx ; 
    const NPY<unsigned>* idx ; 

    std::string desc() const ;
};

struct YOG_API Mt  // material
{
    std::string name ; 
    std::string desc() const ;
};


}  // namespace

