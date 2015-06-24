#pragma once

#include <string>
#include <map>
#include <vector>

#include "glm/fwd.hpp"

class Types {
   public:
       static const char* PHOTONS_ ; 
       static const char* RECORDS_ ; 
       static const char* HISTORY_ ; 
       static const char* MATERIAL_ ; 
       static const char* HISTORYSEQ_ ; 
       static const char* MATERIALSEQ_ ; 

       typedef enum { PHOTONS, RECORDS, HISTORY, MATERIAL, HISTORYSEQ, MATERIALSEQ } Item_t ;

       Types(); 
       std::vector<std::string>& getFlagLabels();
       bool*                     getFlagSelection();


       const char* getItemName(Item_t item);
       glm::ivec4 getFlags();

       std::string getMaskString(unsigned int mask, Item_t etype);
       std::string getMaterialString(unsigned int flags);
       std::string getHistoryString(unsigned int flags);
       std::string getStepFlagString(unsigned char flag);

       bool* initBooleanSelection(unsigned int n);
       void readFlags(const char* path); // parse enum flags from photon.h
       void readMaterials(const char* idpath, const char* name="GMaterialIndexLocal.json");    

       void dumpFlags(const char* msg="PhotonsNPY::dumpFlags");
       void dumpMaterials(const char* msg="PhotonsNPY::dumpMaterials");

       std::string findMaterialName(unsigned int index);

   private:
       std::map<std::string, unsigned int>                  m_materials ;
       //std::vector< std::pair<unsigned int, std::string> >  m_flags ; 
       std::vector<std::string>                             m_flag_labels ; 
       std::vector<unsigned int>                            m_flag_codes ; 
       bool*                                                m_flag_selection ; 


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


