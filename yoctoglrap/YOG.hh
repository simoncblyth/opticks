#pragma once

#include <string>
#include <vector>
#include <map>

#include "YOG_API_EXPORT.hh"
#include <glm/fwd.hpp>

/**
YOG
===========

* ../analytic/sc.py

**/

struct nnode ; 
class GMesh ; 


namespace YOG {

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

struct Sc ; 

struct YOG_API Nd
{
    int              ndIdx ; 
    int              soIdx ;   // mesh 
    const glm::mat4* transform ; 
    std::string      boundary ; 
    std::string      name ;   // pvname 
    int              depth ;
    const Sc*        scene ; 
    bool             selected ; 
    int              parent ; 

    std::vector<int> children ; 
    std::string desc() const ;
};

struct YOG_API Sc 
{
    Sc(int root_=0);

    int  root ; 
    std::vector<Nd*>  nodes ; 
    std::vector<Mh*>  meshes ; 

    std::string desc() const ;

    bool has_mesh(int lvIdx) const ; 
    int lv2so(int lvIdx) const ;

    int add_mesh(int   lvIdx,
                 const std::string& lvName, 
                 const std::string& soName);

    int add_node(int   lvIdx, 
                 const std::string& lvName, 
                 const std::string& pvName, 
                 const std::string& soName, 
                 const glm::mat4* transform, 
                 const std::string& boundary,
                 int   depth, 
                 bool  selected);  

    int add_test_node(int lvIdx);

};


}  // namespace


