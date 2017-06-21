#pragma once

#include <map>
#include <string>
#include <vector>

class Opticks ; 

class GSolid ; 
class GNode ; 
class GItemList ; 


#include "GGEO_API_EXPORT.hh"

/*

GNodeLib
===========

For partial geometries targetnode identifies
the full geometry node traversal index of the root node.
By definition this is zero for full geometry, it is obtained from 
top level assert metadata of the GLTF.

*/


class GGEO_API GNodeLib 
{
        friend class GGeo   ;  // for save 
        friend class GScene ;  // for save 
    public:
        static GNodeLib* load(Opticks* ok, const char* reldir);
        GNodeLib(Opticks* opticks, bool loaded, unsigned targetnode, const char* reldir); 
        std::string desc() const ; 
    private:
        void save() const ;
        void init();
        GItemList*   getPVList(); 
        GItemList*   getLVList(); 
    public:
        unsigned getTargetNodeOffset() const ;
        unsigned getNumPV() const ;
        unsigned getNumLV() const ;
        void add(GSolid*    solid);
        GNode* getNode(unsigned index); 
        GSolid* getSolid(unsigned int index);  
        GSolid* getSolidSimple(unsigned int index);  
        unsigned getNumSolids() const ;
    public:
        const char* getPVName(unsigned int index) const ;
        const char* getLVName(unsigned int index) const ;
    private:
        Opticks*                           m_ok ;  
        bool                               m_loaded ; 
        unsigned                           m_targetnode ; 
        const char*                        m_reldir ; 

        GItemList*                         m_pvlist ; 
        GItemList*                         m_lvlist ; 
    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 
        std::vector<GSolid*>               m_solids ; 

};
 


