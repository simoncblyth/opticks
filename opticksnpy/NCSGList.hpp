#pragma once

#include <vector>
#include "NCSG.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NCSGList 
{
   public:
     NCSGList(const char* csgpath, unsigned verbosity);
     void dump(const char* msg="NCSGList::dump");

   public:
     NCSG*    getTree(unsigned index);
     unsigned getNumTrees();

   private:
      const char*        m_csgpath ; 
      unsigned           m_verbosity ; 
      std::vector<NCSG*> m_trees ; 

};
 
