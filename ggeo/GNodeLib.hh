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


*/


class GGEO_API GNodeLib 
{
        friend class GGeo   ;  // for save 
        friend class GScene ;  // for save 
    public:
        static const char* GetRelDir(bool analytic);
        static GNodeLib* Load(Opticks* ok, bool analytic);
        void loadFromCache();
    public:
        GNodeLib(Opticks* opticks, bool analytic); 
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
        const char*                        m_reldir ; 

        GItemList*                         m_pvlist ; 
        GItemList*                         m_lvlist ; 
        GTreePresent*                      m_treepresent ; 
    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 
        std::vector<GSolid*>               m_solids ; 

};
 


