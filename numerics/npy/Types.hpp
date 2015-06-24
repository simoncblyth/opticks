#pragma once

#include <string>
#include <map>
#include <vector>

#include "glm/fwd.hpp"

class Types {
   public:
       static const char* HISTORY_ ; 
       static const char* MATERIAL_ ; 
       static const char* HISTORYSEQ_ ; 
       static const char* MATERIALSEQ_ ; 

       typedef enum { HISTORY, MATERIAL, HISTORYSEQ, MATERIALSEQ } Item_t ;

       Types(); 

       const char* getItemName(Item_t item);
   public:
       std::string getMaskString(unsigned int mask, Item_t etype, bool abbrev=false);

   public:
       std::string getMaterialString(unsigned int flags, bool abbrev=false);
       void readMaterials(const char* idpath, const char* name="GMaterialIndexLocal.json");    
       void dumpMaterials(const char* msg="PhotonsNPY::dumpMaterials");
       std::string findMaterialName(unsigned int index);
   private:
       void makeMaterialAbbrev();

   public:
       std::string getStepFlagString(unsigned char flag);
       std::string getHistoryString(unsigned int flags, bool abbrev=false, const char* tail=" ");
       void readFlags(const char* path); // parse enum flags from photon.h
       void dumpFlags(const char* msg="PhotonsNPY::dumpFlags");
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
       std::map<std::string, unsigned int>                  m_materials ;
       std::map<std::string, std::string>                   m_material2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2material ; 

   private:
       std::vector<std::string>                             m_flag_labels ; 
       std::vector<unsigned int>                            m_flag_codes ; 
       bool*                                                m_flag_selection ; 
   private:
       std::map<std::string, std::string>                   m_flag2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2flag ; 


};


inline Types::Types()
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


