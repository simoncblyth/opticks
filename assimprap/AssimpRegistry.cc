/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>

#include "AssimpRegistry.hh"
#include "AssimpNode.hh"

#include "PLOG.hh"
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
   LOG(info) << msg 
             << " size " << m_registry.size()
             ;
}


