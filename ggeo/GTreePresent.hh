#pragma once

#include <string>
#include <vector>

class GNode ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"


/**
GTreePresent
=============

Creates a text dump of a volume tree with suppression 
of large numbers of siblings to make more readable.

Removed GGeo dependency, now pass in top node at use::

    GNode* top = m_ggeo->getSolid(0); 
    treepresent->traverse(top)

Compare with env/geant4/geometry/collada/g4daenode.py DAESubTree

**/


class GGEO_API GTreePresent 
{
        static const char* NONAME ; 
   public:
        GTreePresent(unsigned int depth_max, unsigned int sibling_max );
   public:
        void traverse(GNode* top);
        void dump(const char* msg="GTreePresent::dump");
        void write(const char* path, const char* reldir);
   private:
        void traverse( GNode* node, unsigned int depth, unsigned int numSibling, unsigned int siblingIndex, bool elide);
   private:
       unsigned int             m_depth_max ; 
       unsigned int             m_sibling_max ; 
   private:
       std::vector<std::string> m_flat ; 
 
};

#include "GGEO_TAIL.hh"


