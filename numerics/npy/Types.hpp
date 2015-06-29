#pragma once

#include <string>
#include <map>
#include <vector>
#include "string.h"

#include "glm/fwd.hpp"

class Index ; 


class Types {
   public:
       static const char* TAIL ; 

       static const char* HISTORY_ ; 
       static const char* MATERIAL_ ; 
       static const char* HISTORYSEQ_ ; 
       static const char* MATERIALSEQ_ ; 

       typedef enum { HISTORY, MATERIAL, HISTORYSEQ, MATERIALSEQ } Item_t ;

   public:
       Types(); 

       void setTail(const char* tail);
       const char* getTail();
       void setAbbrev(bool abbrev);

   public:
       Index*                    getFlagsIndex();
       Index*                    getMaterialsIndex();
       void                      setMaterialsIndex(Index* index);
   public:
       std::string getMaskString(unsigned int mask, Item_t etype);
       std::string getMaterialString(unsigned int flags);
       std::string getHistoryString(unsigned int flags);
       std::string getSequenceString(unsigned long long seq);
       const char* getItemName(Item_t item);

   public:
       void getMaterialStringTest();
       void readMaterials(const char* idpath, const char* name="GMaterialIndexLocal.json");    
       void dumpMaterials(const char* msg="Types::dumpMaterials");
       std::string findMaterialName(unsigned int index);
   private:
       void makeMaterialAbbrev();

   public:
       std::string getStepFlagString(unsigned char flag);
       void getHistoryStringTest();

       void readFlags(const char* path); // parse enum flags from photon.h
       void dumpFlags(const char* msg="Types::dumpFlags");
   private:
       void makeFlagAbbrev();
   public:
       glm::ivec4                getFlags();
       std::vector<std::string>& getFlagLabels();
       bool*                     initBooleanSelection(unsigned int n);
       bool*                     getFlagSelection();
   public:
       std::string getHistoryAbbrev(std::string label);
       std::string getMaterialAbbrev(std::string label);
       std::string getAbbrev(std::string label, Item_t etype);
   public:
       std::string getHistoryAbbrevInvert(std::string label);
       std::string getMaterialAbbrevInvert(std::string label);
       std::string getAbbrevInvert(std::string label, Item_t etype);


   private:
       Index*                                               m_materials_index ; 
       std::map<std::string, unsigned int>                  m_materials ;
       std::map<std::string, std::string>                   m_material2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2material ; 

   private:
       Index*                                               m_flags ; 
       std::vector<std::string>                             m_flag_labels ; 
       std::vector<unsigned int>                            m_flag_codes ; 
       bool*                                                m_flag_selection ; 
   private:
       std::map<std::string, std::string>                   m_flag2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2flag ; 

   private:
       const char*  m_tail ;
       bool         m_abbrev ; 

};


inline Types::Types() :
     m_materials_index(NULL),
     m_flags(NULL),
     m_tail(TAIL),
     m_abbrev(false)
{
};


inline std::vector<std::string>& Types::getFlagLabels()
{
     return m_flag_labels ; 
}
inline bool* Types::getFlagSelection()
{
     return m_flag_selection ; 
}
inline Index* Types::getFlagsIndex()
{
     return m_flags ; 
}


inline void Types::setTail(const char* tail)
{
     m_tail = strdup(tail);
}
inline void Types::setAbbrev(bool abbrev)
{
     m_abbrev = abbrev ;
}
inline const char* Types::getTail()
{
    return m_tail ; 
}



inline Index* Types::getMaterialsIndex()
{
    return m_materials_index ; 
}

