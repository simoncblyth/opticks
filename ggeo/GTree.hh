#pragma once

#include <vector>

class GNode ; 
class GSolid ; 

template<class T> class NPY ;

/**
GTree
=============

Pulling out intended-to-be-common parts of GScene and GTreeCheck into GTree
to avoid duplicity issues. 


**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GTree {
   public: 
       static NPY<float>*    makeInstanceTransformsBuffer(const std::vector<GNode*>& placements);
       static NPY<unsigned>* makeAnalyticInstanceIdentityBuffer(const std::vector<GNode*>& placements)  ;
       static NPY<unsigned>* makeInstanceIdentityBuffer(const std::vector<GNode*>& placements)  ;

};





