#pragma once

#include <map>
#include <vector>

class Opticks ; 

class GSolid ; 
class GNode ; 
class GItemList ; 


#include "GGEO_API_EXPORT.hh"

class GGEO_API GNodeLib {

        friend class GGeo ;  // for save 
    public:
        static GNodeLib* load(Opticks* ok);
        GNodeLib(Opticks* opticks, bool loaded); 
    private:
        void save();
        void init();
        GItemList*   getPVList(); 
        GItemList*   getLVList(); 
    public:
        unsigned getNumPV();
        unsigned getNumLV();
        void add(GSolid*    solid);
        GNode* getNode(unsigned index); 
        GSolid* getSolid(unsigned int index);  
        GSolid* getSolidSimple(unsigned int index);  
        unsigned getNumSolids();
    public:
        const char* getPVName(unsigned int index);
        const char* getLVName(unsigned int index);


    private:
        Opticks*                           m_ok ;  
        bool                               m_loaded ; 
        GItemList*                         m_pvlist ; 
        GItemList*                         m_lvlist ; 
    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 
        std::vector<GSolid*>               m_solids ; 

};
 


