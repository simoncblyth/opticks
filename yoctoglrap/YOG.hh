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

namespace YOG {

struct YOG_API Mh
{
    int         lvIdx ; 
    std::string lvName ; 
    std::string soName ;
    int         soIdx ; 

    std::string desc() const ;
};

struct Sc ; 

struct YOG_API Nd
{
    int              ndIdx ; 
    int              soIdx ; 
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
    std::vector<Nd*>  nodes ; 
    std::vector<Mh*>  meshes ; 

    std::string desc() const ;

    bool has_mesh(int lvIdx) const ; 

    int add_mesh(int lvIdx,
                 const std::string& lvName, 
                 const std::string& soName);


    int lv2so(int lvIdx) const ;

    int add_node(int lvIdx, 
                 const std::string& lvName, 
                 const std::string& pvName, 
                 const std::string& soName, 
                 const glm::mat4* transform, 
                 const std::string& boundary,
                 int depth, 
                 bool selected);  

};


}  // namespace


