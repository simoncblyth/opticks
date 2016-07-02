#pragma once

#include <map>
class AssimpNode ; 

#include "ASIRAP_API_EXPORT.hh"
#include "ASIRAP_HEAD.hh"

class ASIRAP_API AssimpRegistry {
public:
   AssimpRegistry();
   virtual ~AssimpRegistry();

public:
   void add(AssimpNode* node);
   AssimpNode* lookup(std::size_t hash);  
   void summary(const char* msg="AssimpRegistry::summary");

private:
   std::map<std::size_t, AssimpNode*> m_registry ;  
};

#include "ASIRAP_TAIL.hh"


