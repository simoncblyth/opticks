#pragma once

#include <map>
#include <string>
#include <vector>

class Opticks ; 

class GSolid ; 
class GNode ; 
class GItemList ; 

class GTreePresent ; 


#include "GGEO_API_EXPORT.hh"

/*

GNodeLib
===========

Collection of GSolid/GNode instances with access by index.
NB only pv/lv names are persisted, not the solids/nodes.
Initially primarily a pre-cache operator, but access to pv/lv names also 
relevant post-cache.

There are several canonical m_nodelib instances:

*GGeo::init precache non-analytic*

     874 void GGeo::add(GSolid* solid)
     875 {
     876     m_nodelib->add(solid);
     877 }


*GScene::GScene analytic*

     m_nodelib(loaded ? GNodeLib::Load(m_ok, m_analytic ) : new GNodeLib(m_ok, m_analytic)), 

     893 void GScene::addNode(GSolid* node, nd* n)
     894 {
     895     unsigned node_idx = n->idx ;
     896     assert(m_nodes.count(node_idx) == 0);
     897     m_nodes[node_idx] = node ;
     898 
     899     // TODO ... get rid of above, use the nodelib 
     900     m_nodelib->add(node);
     901 }

*/

class GGEO_API GNodeLib 
{
        friend class GGeo   ;  // for save 
        friend class GScene ;  // for save 
    public:
        static const char* GetRelDir(bool analytic, bool test);
        static GNodeLib* Load(Opticks* ok, bool analytic, bool test);
        void loadFromCache();
    public:
        GNodeLib(Opticks* opticks, bool analytic, bool test); 
        std::string desc() const ; 
    private:
        void save() const ;
        void init();
        GItemList*   getPVList(); 
        GItemList*   getLVList(); 
    public:
        //unsigned getTargetNodeOffset() const ;
        unsigned getNumPV() const ;
        unsigned getNumLV() const ;
        void add(GSolid*    solid);
        GNode* getNode(unsigned index) const ; 
        GSolid* getSolid(unsigned int index) const ;  
        GSolid* getSolidSimple(unsigned int index);  
        unsigned getNumSolids() const ;
    public:
        const char* getPVName(unsigned int index) const ;
        const char* getLVName(unsigned int index) const ;
    private:
        Opticks*                           m_ok ;  
        bool                               m_analytic ; 
        bool                               m_test ; 
        const char*                        m_reldir ; 

        GItemList*                         m_pvlist ; 
        GItemList*                         m_lvlist ; 
        GTreePresent*                      m_treepresent ; 
    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 
        std::vector<GSolid*>               m_solids ; 
};
 

