#pragma once
/**
MTool
========

OpenMesh machinery for mesh traversal and mending cleaved meshes.


**/


#include <string>

class GMesh ; 
class Opticks ; 

#include "MESHRAP_API_EXPORT.hh"
#include "MESHRAP_HEAD.hh"
class MESHRAP_API MTool {
   public:
       MTool();
       static GMesh* joinSplitUnion(GMesh* mesh, Opticks* opticks);
   public:
       unsigned int countMeshComponents(const GMesh* gm);
   public:
       std::string& getOut();
       std::string& getErr();
       unsigned int getNoise();
   private:
       unsigned int countMeshComponents_(const GMesh* gm);

   private:
       std::string  m_out ; 
       std::string  m_err ;
       unsigned int m_noise ;  

};

#include "MESHRAP_TAIL.hh"

