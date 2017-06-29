#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>
#include <string>

#include <glm/fwd.hpp>

struct nd ; 
struct nmat4triple ;

#include "NBBox.hpp"

/*

nd : structural nodes from glTF
===================================

* nd kept minimal, most use via NScene 
  
  * nd are populated from the glTF by NScene::import_r


*/


struct NPY_API nd
{
   unsigned         idx ;
   int              repeatIdx ;

   unsigned         mesh ; 
   unsigned         depth ; 
   std::string      boundary ; 
   std::string      pvname ; 
   unsigned         selected ; 
   unsigned         containment ; 

   std::string      _local_digest ; 
   std::string      _mesh_digest ; 
   std::string      _progeny_digest ; 

   nd*              parent ; 
   nmat4triple*     transform ; 
   nmat4triple*     gtransform ; 
   std::vector<nd*> children ; 
   std::vector<nd*> _progeny ; 
   std::vector<nd*> _ancestors ; 


   nbbox                  aabb ;  
   std::vector<glm::vec3> surf ; 


   std::string desc();
   std::string detail();
   static nmat4triple* make_global_transform(nd* n) ; 


   static std::string      _make_digest(const std::vector<nd*>& nds, nd* extra);
   static void             _collect_progeny_r(nd* n, std::vector<nd*>& progeny, int depth);

   std::string             _make_local_digest() const ;
   std::string             _make_mesh_digest() const ;
   std::string             _mesh_id() const ;

   void                    _collect_nodes_r(std::vector<nd*>& selection, const std::string& pdig) ;
   nd*                     _find_node_r(const std::string& pdig) ;
   void                    _collect_ancestors(nd* n, std::vector<nd*>& ancestors) ;


   std::string             pvtag(unsigned nch=30) const ; 

   unsigned                get_progeny_count(); 
   const std::vector<nd*>& get_progeny(); 
   const std::vector<nd*>& get_ancestors(); 
   const std::string&      get_local_digest(); 
   const std::string&      get_mesh_digest(); 
   const std::string&      get_progeny_digest(); 
   bool                    has_progeny_digest(const std::string& dig);

   std::vector<nd*>        find_nodes(const std::string& pdig) ; // all nodes with the progeny digest   
   nd*                     find_node(const std::string& pdig) ;  // first node with the progeny digest

};


