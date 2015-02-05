#ifndef ASSIMPREGISTRY_H
#define ASSIMPREGISTRY_H

#include <map>
class AssimpNode ; 

class AssimpRegistry {
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


#endif

