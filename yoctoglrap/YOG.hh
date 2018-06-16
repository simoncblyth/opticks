#pragma once

#include <string>
#include <vector>

#include "YOG_API_EXPORT.hh"
#include <glm/fwd.hpp>

/**
YOG : intermediate geometry tree
===================================

The YOG geometry model (Sc, Nd, Mh) is used as an intermediary model 
at a slightly higher level than ygltf, just above the gltf details.
The Geant4 volume tree is converted into YOG by calling YOG::Nd::add_node
from within the recursive X4PhysicalVolume::TraverseVolumeTree.
Subsequently Sc can be converted to ygltf and saved to file 
using YOG::TF.

The YOG model follows that provided in ../analytic/sc.py
which was used to convert a parsed GDML into non-renderable
GLTF for consumption by NGLTF/NScene.

**/

struct nnode ; 
class GMesh ; 


namespace YOG {

struct Nd ; 
struct Mh ; 

struct YOG_API Sc 
{
    Sc(int root_=0);

    int  root ; 
    std::vector<Nd*>  nodes ; 
    std::vector<Mh*>  meshes ; 

    std::string desc() const ;

    int add_node(int   lvIdx, 
                 const std::string& lvName, 
                 const std::string& pvName, 
                 const std::string& soName, 
                 const glm::mat4* transform, 
                 const std::string& boundary,
                 int   depth, 
                 bool  selected);  

    int add_test_node(int lvIdx);

    int lv2so(int lvIdx) const ;
    bool has_mesh(int lvIdx) const ; 
    int add_mesh(int   lvIdx,
                 const std::string& lvName, 
                 const std::string& soName);

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
    int         lvIdx ; 
    std::string lvName ; 
    std::string soName ;
    int         soIdx ; 
    nnode*      csg ; 
    GMesh*      mesh ; 

    std::string desc() const ;
};


}  // namespace


