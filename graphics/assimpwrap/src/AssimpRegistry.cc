#include "AssimpRegistry.hh"
#include "AssimpNode.hh"
#include "assert.h"
#include "stdio.h"


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


AssimpRegistry::AssimpRegistry()
{
}
AssimpRegistry::~AssimpRegistry()
{
}

void AssimpRegistry::add(AssimpNode* node)
{
   std::size_t digest = node->getDigest();
   AssimpNode* prior = lookup(digest);
   assert(!prior);
   m_registry[digest] = node ;
}

AssimpNode* AssimpRegistry::lookup(std::size_t digest)
{
   return m_registry.find(digest) != m_registry.end() ? m_registry[digest] : NULL ; 
}  

void AssimpRegistry::summary(const char* msg)
{
   printf("%s size %lu \n", msg, m_registry.size() );
}


