#include <glm/glm.hpp>
#include "Types.hpp"
#include "jsonutil.hpp" 
#include "regexsearch.hh"
#include <sstream>


const char* Types::PHOTONS_ = "photons" ;
const char* Types::RECORDS_ = "records" ;
const char* Types::HISTORY_ = "history" ;
const char* Types::MATERIAL_ = "material" ;
const char* Types::HISTORYSEQ_ = "historyseq" ;
const char* Types::MATERIALSEQ_ = "materialseq" ;


const char* Types::getItemName(Item_t item)
{
    const char* name(NULL);
    switch(item)
    {
       case PHOTONS:name = PHOTONS_ ; break ; 
       case RECORDS:name = RECORDS_ ; break ; 
       case HISTORY:name = HISTORY_ ; break ; 
       case MATERIAL:name = MATERIAL_ ; break ; 
       case HISTORYSEQ:name = HISTORYSEQ_ ; break ; 
       case MATERIALSEQ:name = MATERIALSEQ_ ; break ; 
    }
    return name ; 
}


void Types::readMaterials(const char* idpath, const char* name)
{
    loadMap<std::string, unsigned int>(m_materials, idpath, name);
}
void Types::dumpMaterials(const char* msg)
{
    dumpMap<std::string, unsigned int>(m_materials, msg);
}

std::string Types::findMaterialName(unsigned int index)
{
    std::stringstream ss ; 
    ss << "findMaterialName-failed-" << index  ;
    std::string name = ss.str() ;
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        if( it->second == index )
        {
            name = it->first ;
            break ; 
        }
    }
    return name ; 
}

std::string Types::getMaterialString(unsigned int mask)
{
    std::stringstream ss ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        unsigned int mat = it->second ;
        if(mask & (1 << (mat-1))) ss << it->first << " " ; 
    }
    return ss.str() ; 
}

std::string Types::getHistoryString(unsigned int flags)
{
    std::stringstream ss ; 
    for(unsigned int i=0 ; i < m_flag_labels.size() ; i++)
    {
        unsigned int mask = m_flag_codes[i] ;
        if(flags & mask) ss << m_flag_labels[i] << " " ; 
    }
    return ss.str() ; 
}

std::string Types::getStepFlagString(unsigned char flag)
{
   return getHistoryString( 1 << (flag-1) ); 
}




std::string Types::getMaskString(unsigned int mask, Item_t etype)
{
    std::string mstr ;
    switch(etype)
    {
       case HISTORY:mstr = getHistoryString(mask) ; break ; 
       case MATERIAL:mstr = getMaterialString(mask) ; break ; 
       case PHOTONS:mstr = "??" ; break ; 
       case RECORDS:mstr = "??" ; break ; 
       case MATERIALSEQ:mstr = "??" ; break ; 
       case HISTORYSEQ:mstr = "??" ; break ; 
    }
    return mstr ; 
}





void Types::readFlags(const char* path)
{
    // read photon header to get flag names and enum values

    typedef std::pair<unsigned int, std::string>  upair_t ;
    typedef std::vector<upair_t>                  upairs_t ;
    upairs_t ups ; 
    enum_regexsearch( ups, path ); // "$ENV_HOME/graphics/ggeoview/cu/photon.h");    

    m_flag_labels.clear();
    m_flag_codes.clear();
    for(unsigned int i=0 ; i < ups.size() ; i++)
    {   
        upair_t p = ups[i];
        m_flag_codes.push_back(p.first);         
        m_flag_labels.push_back(p.second);         
    }   
    m_flag_selection = initBooleanSelection(m_flag_labels.size());
}



bool* Types::initBooleanSelection(unsigned int n)
{
    bool* selection = new bool[n];
    while(n--) selection[n] = false ; 
    return selection ;
}





void Types::dumpFlags(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_flag_labels.size() ; i++)
    {
         unsigned int code = m_flag_codes[i] ;
         std::string label = m_flag_labels[i] ;
         printf(" %10d : %10x :  %s  : %d \n", code, code,  label.c_str(), m_flag_selection[i] );
    }
}


glm::ivec4 Types::getFlags()
{
    int flags(0) ;
    for(unsigned int i=0 ; i < m_flag_labels.size() ; i++)
    {
        if(m_flag_selection[i]) flags |= m_flag_codes[i] ; 
    } 
    return glm::ivec4(flags,0,0,0) ;     
}



