#include <glm/glm.hpp>
#include "Types.hpp"
#include "jsonutil.hpp" 
#include "regexsearch.hh"
#include <sstream>
#include <iostream>
#include <iomanip>


const char* Types::HISTORY_ = "history" ;
const char* Types::MATERIAL_ = "material" ;
const char* Types::HISTORYSEQ_ = "historyseq" ;
const char* Types::MATERIALSEQ_ = "materialseq" ;


const char* Types::getItemName(Item_t item)
{
    const char* name(NULL);
    switch(item)
    {
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
    makeMaterialAbbrev();
}

void Types::makeMaterialAbbrev()
{
    typedef std::map<std::string, unsigned int> MSU ; 
    typedef std::map<std::string, std::string>  MSS ; 

    // special cases where 1st 2-chars not unique
    MSS s ; 
    s["NitrogenGas"] = "NG" ;   

    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        std::string mat = it->first ;
        std::string abb = s.count(mat) == 1 ? s[mat] : mat.substr(0,2);
        m_material2abbrev[mat] = abb ;
        m_abbrev2material[abb] = mat ;
    }
    assert(m_material2abbrev.size() == m_abbrev2material.size());
}


std::string Types::getMaterialAbbrev(std::string label)
{
     return m_material2abbrev.count(label) == 1 ? m_material2abbrev[label] : label  ;
}
std::string Types::getMaterialAbbrevInvert(std::string label)
{
     return m_abbrev2material.count(label) == 1 ? m_abbrev2material[label] : label  ;
}



void Types::dumpMaterials(const char* msg)
{
    //dumpMap<std::string, unsigned int>(m_materials, msg);
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        unsigned int code = it->second ; 
        std::string mat = it->first ;
        std::string abb = getMaterialAbbrev(mat);

        std::cout 
              << std::setw(3) << code 
              << std::setw(5) << abb 
              << std::setw(25) << mat 
              << std::endl ; 
    }
 

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

std::string Types::getMaterialString(unsigned int mask, bool abbrev)
{
    std::stringstream ss ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        unsigned int mat = it->second ;
        if(mask & (1 << (mat-1))) 
        {
            std::string label = it->first ; 
            if(abbrev) label = getMaterialAbbrev(label) ;
            ss << label << " " ; 
        }
    }
    return ss.str() ; 
}

std::string Types::getHistoryAbbrev(std::string label)
{
     return m_flag2abbrev.count(label) == 1 ? m_flag2abbrev[label] : label  ;
}
std::string Types::getHistoryAbbrevInvert(std::string label)
{
     return m_abbrev2flag.count(label) == 1 ? m_abbrev2flag[label] : label  ;
}


std::string Types::getAbbrev(std::string label, Item_t etype)
{
    std::string abb ;
    switch(etype)
    {
        case HISTORY :abb = getHistoryAbbrev(label) ; break ; 
        case MATERIAL:abb = getMaterialAbbrev(label) ; break ; 
        case HISTORYSEQ:   ;break; 
        case MATERIALSEQ:  ;break; 
    }
    return abb ; 
}

std::string Types::getAbbrevInvert(std::string label, Item_t etype)
{
    std::string abb ;
    switch(etype)
    {
        case HISTORY :abb = getHistoryAbbrevInvert(label) ; break ; 
        case MATERIAL:abb = getMaterialAbbrevInvert(label) ; break ; 
        case HISTORYSEQ:   ;break; 
        case MATERIALSEQ:  ;break; 
    }
    return abb ; 
}


std::string Types::getHistoryString(unsigned int flags, bool abbrev, const char* tail)
{
    std::stringstream ss ; 
    for(unsigned int i=0 ; i < m_flag_labels.size() ; i++)
    {
        unsigned int mask = m_flag_codes[i] ;
        if(flags & mask)
        {
            std::string label = m_flag_labels[i] ;
            if(abbrev) label = getHistoryAbbrev(label) ;
            ss << label << tail  ; 
        }
    }
    return ss.str() ; 
}



std::string Types::getStepFlagString(unsigned char flag)
{
   return getHistoryString( 1 << (flag-1) ); 
}




std::string Types::getMaskString(unsigned int mask, Item_t etype, bool abbrev )
{
    std::string mstr ;
    switch(etype)
    {
       case HISTORY:mstr = getHistoryString(mask, abbrev) ; break ; 
       case MATERIAL:mstr = getMaterialString(mask, abbrev) ; break ; 
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

    makeFlagAbbrev();
}

void Types::makeFlagAbbrev()
{
    typedef std::map<std::string,std::string> MSS ; 
    MSS s ; 
    s["CERENKOV"]          = "CE" ;
    s["SCINTILLATION"]     = "SC" ;
    s["MISS"]              = "MI" ;
    s["BULK_ABSORB"]       = "KA" ;
    s["BULK_REEMIT"]       = "KR" ;
    s["BULK_SCATTER"]      = "KS" ;
    s["SURFACE_DETECT"]    = "SD" ;
    s["SURFACE_ABSORB"]    = "SA" ;
    s["SURFACE_DREFLECT"]  = "DR" ;
    s["SURFACE_SREFLECT"]  = "SR" ;
    s["BOUNDARY_REFLECT"]  = "BR" ;  
    s["BOUNDARY_TRANSMIT"] = "BT" ;
    s["NAN_ABORT"]         = "NA" ;

    for(MSS::iterator it=s.begin() ; it!=s.end() ; it++)
    {
        m_flag2abbrev[it->first] = it->second ; 
        m_abbrev2flag[it->second] = it->first ; 

    }
    assert(m_flag2abbrev.size() == m_abbrev2flag.size()); 
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
         std::string check = getHistoryString(code, false, "");
         if(strcmp(label.c_str(),check.c_str()) != 0 )
         {
             printf("mismatch\n");
         }
         std::string abbrev = getHistoryAbbrev(label) ;
         printf(" %10d : %10x : %20s : %20s :  %2s :  %d \n", code, code, label.c_str(), check.c_str(), abbrev.c_str(), m_flag_selection[i] );
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



