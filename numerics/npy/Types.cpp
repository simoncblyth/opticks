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
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
        std::pair<unsigned int, std::string> p = m_flags[i];
        unsigned int mask = p.first ;
        if(flags & mask) ss << p.second << " " ; 
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
    enum_regexsearch( m_flags, path);
    m_flags_selection = initBooleanSelection(m_flags.size());
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
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
         std::pair<unsigned int, std::string> p = m_flags[i];
         printf(" %10d : %10x :  %s  : %d \n", p.first, p.first,  p.second.c_str(), m_flags_selection[i] );
    }
}


glm::ivec4 Types::getFlags()
{
    int flags(0) ;
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
        if(m_flags_selection[i]) flags |= m_flags[i].first ; 
    } 
    return glm::ivec4(flags,0,0,0) ;     
}






