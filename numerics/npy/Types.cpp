#include <glm/glm.hpp>
#include "Types.hpp"
#include "Index.hpp"
#include "jsonutil.hpp" 
#include "stringutil.hpp" 
#include "regexsearch.hh"

#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Types::TAIL = " " ;

const char* Types::HISTORY_ = "history" ;
const char* Types::MATERIAL_ = "material" ;
const char* Types::HISTORYSEQ_ = "historyseq" ;
const char* Types::MATERIALSEQ_ = "materialseq" ;


void Types::init()
{
    readFlags("$ENV_HOME/graphics/optixrap/cu/photon.h");  
}

void Types::saveFlags(const char* idpath, const char* ext)
{
    m_flags->setExt(ext); 
    m_flags->save(idpath);    
}


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


void Types::readMaterialsOld(const char* idpath, const char* name)
{
    typedef std::map<std::string, unsigned int> MSU ; 

    MSU mats ; 
    loadMap<std::string, unsigned int>(mats, idpath, name);
    for(MSU::iterator it=mats.begin() ; it != mats.end() ; it++)
    {
        m_materials[it->first + m_tail] = it->second ; 
    } 
    makeMaterialAbbrev();
}

void Types::readMaterials(const char* idpath, const char* name)
{
    Index* index = Index::load( idpath, name );
    setMaterialsIndex(index);
}


void Types::setMaterialsIndex(Index* index)
{
    m_materials_index = index ; 

    typedef std::vector<std::string> VS ; 
    VS& names = index->getNames();

    for(VS::iterator it=names.begin() ; it != names.end() ; it++)
    {
        std::string name = *it ; 
        unsigned int local = index->getIndexLocal(name.c_str()) ;
        m_materials[name + m_tail] = local; 
    }
    makeMaterialAbbrev();
}


void Types::makeMaterialAbbrev()
{
    typedef std::map<std::string, unsigned int> MSU ; 
    typedef std::map<std::string, std::string>  MSS ; 

    // special cases where 1st 2-chars not unique or misleading
    MSS s ; 
    s["NitrogenGas "] = "NG" ;   
    s["ADTableStainlessSteel "] = "TS" ;
    s["LiquidScintillator "] = "Ls" ; 
    s["MineralOil "] = "MO" ; 
    s["StainlessSteel "] = "SS" ;  
    // NB must have a single white space after the material name to match names from m_materials

    // Hmm not very sustainable with multiple detctors, 
    // have to play around to avoid non-unique names 
    // TODO: make the abbrevs an input 
    //
    // NB the tail spacer, to match what is done in readMaterials

    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        std::string mat = it->first ;
        std::string abb = s.count(mat) == 1 ? s[mat] : mat.substr(0,2) ;
        abb += m_tail ;  

        if(m_material2abbrev.count(mat) > 0)
        {
            printf("Types::makeMaterialAbbrev ambiguous material name %s \n", mat.c_str());
        } 

        if(m_abbrev2material.count(abb) > 0)
        {
            printf("Types::makeMaterialAbbrev ambiguous material abbrev %s \n", abb.c_str());
        } 

        //printf("Types::makeMaterialAbbrev [%s] [%s] \n", mat.c_str(), abb.c_str() );

        m_material2abbrev[mat] = abb ;
        m_abbrev2material[abb] = mat ;
    }
    assert(m_material2abbrev.size() == m_abbrev2material.size());
}


std::string Types::getMaterialAbbrev(std::string label)
{
     return m_material2abbrev.count(label) == 1 ? m_material2abbrev[label] : label  ;
}
std::string Types::getMaterialAbbrevInvert(std::string label, bool hex)
{
     if(hex)
     {
         std::string name = m_materials_index ? m_materials_index->getNameLocal(hex_lexical_cast<unsigned int>(label.c_str()), "?" ) : label ;
         return name + m_tail ; 
     }
     return m_abbrev2material.count(label) == 1 ? m_abbrev2material[label] : label  ;
}

unsigned int Types::getMaterialAbbrevInvertAsCode(std::string label, bool hex)
{
     if(hex) return hex_lexical_cast<unsigned int>(label.c_str()) ;

     unsigned int n = m_abbrev2material.count(label) ;
     assert(n == 1);

     std::string matn = m_abbrev2material[label];
     return getMaterialCode(matn);
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
              << "[" << abb << "] " 
              << "[" << mat << "] "  
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

std::string Types::getMaterialString(unsigned int mask)
{
    std::stringstream ss ; 
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_materials.begin() ; it != m_materials.end() ; it++)
    {
        unsigned int mat = it->second ;
        if(mask & (1 << (mat-1))) 
        {
            std::string label = it->first ; 
            if(m_abbrev) label = getMaterialAbbrev(label) ;
            ss << label ; //  << m_tail ; 
        }
    }
    return ss.str() ; 
}


unsigned int Types::getMaterialCode(std::string label)
{
    std::string labeltt = label.substr(0,label.size()-1);  // trim tail spacer
    assert(m_materials_index);
    return m_materials_index->getIndexSource(labeltt.c_str(), 0xFF);    
}


void Types::getMaterialStringTest()
{
    for(unsigned int i=0 ; i < 32 ; i++)
    {
        unsigned int mask = 1 << i ; 
        std::string label = getMaterialString( mask );
        std::string abbrev = getMaterialAbbrev( label );
        unsigned int code = getMaterialCode(label);

        printf("%3d : %8x : [%s] [%s] code[%u] \n", i, mask, label.c_str(), abbrev.c_str(), code ); 
    }
}


std::string Types::getHistoryAbbrev(std::string label)
{
     return m_flag2abbrev.count(label) == 1 ? m_flag2abbrev[label] : label  ;
}
std::string Types::getHistoryAbbrevInvert(std::string label, bool hex)
{
     if(hex)
     {
         std::string name = m_flags->getNameLocal(hex_lexical_cast<unsigned int>(label.c_str()), "?" );
         return name + m_tail ; 
     } 

     unsigned int n = m_abbrev2flag.count(label) ;
    // printf("Types::getHistoryAbbrevInvert [%s] %u \n", label.c_str(), n );  
     return n == 1 ? m_abbrev2flag[label] : label  ;
}

unsigned int Types::getHistoryAbbrevInvertAsCode(std::string label, bool hex)
{
     if(hex) return hex_lexical_cast<unsigned int>(label.c_str()) ;

     unsigned int n = m_abbrev2flag.count(label) ;
    // printf("Types::getHistoryAbbrevInvert [%s] %u \n", label.c_str(), n );  
     assert(n == 1);

     std::string flag = m_abbrev2flag[label];
     return getHistoryFlag(flag);
}






std::string Types::getAbbrev(std::string label, Item_t etype)
{
    std::string abb ;
    switch(etype)
    {
        case HISTORY     : abb = getHistoryAbbrev(label)  ; break ; 
        case HISTORYSEQ  : abb = getHistoryAbbrev(label)  ; break ; 
        case MATERIAL    : abb = getMaterialAbbrev(label) ; break ; 
        case MATERIALSEQ : abb = getMaterialAbbrev(label) ; break ; 
    }
    return abb ; 
}

std::string Types::getAbbrevInvert(std::string label, Item_t etype, bool hex)
{
    std::string abb ;
    switch(etype)
    {
        case HISTORY     : abb = getHistoryAbbrevInvert(label, hex)  ; break ; 
        case HISTORYSEQ  : abb = getHistoryAbbrevInvert(label, hex)  ; break ; 
        case MATERIAL    : abb = getMaterialAbbrevInvert(label, hex) ; break ; 
        case MATERIALSEQ : abb = getMaterialAbbrevInvert(label, hex) ; break ; 
    }
    return abb ; 
}


unsigned int Types::getAbbrevInvertAsCode(std::string label, Item_t etype, bool hex)
{
    unsigned int bpos ; 
    switch(etype)
    {
        case HISTORY     : bpos = getHistoryAbbrevInvertAsCode(label, hex)  ; break ; 
        case HISTORYSEQ  : bpos = getHistoryAbbrevInvertAsCode(label, hex)  ; break ; 
        case MATERIAL    : bpos = getMaterialAbbrevInvertAsCode(label, hex) ; break ; 
        case MATERIALSEQ : bpos = getMaterialAbbrevInvertAsCode(label, hex) ; break ; 
    }
    return bpos ; 
}


unsigned int Types::getHistoryFlag(std::string label)
{
    std::string labeltt = label.substr(0,label.size()-1);  // trim tail spacer

    return m_flags->getIndexSource(labeltt.c_str(), 0xFF);    
}



std::string Types::getHistoryString(unsigned int flags)
{
    std::stringstream ss ; 
    for(unsigned int i=0 ; i < m_flag_labels.size() ; i++)
    {
        unsigned int mask = m_flag_codes[i] ;
        if(flags & mask)
        {
            std::string label = m_flag_labels[i] ;
            if(m_abbrev) label = getHistoryAbbrev(label) ;
            ss << label ; // << m_tail  ;  labels now have the tail appended on reading in
        }
    }
    return ss.str() ; 
}

void Types::getHistoryStringTest()
{
    for(unsigned int i=0 ; i < 32 ; i++)
    {
        std::string label = getHistoryString( 1 << i);
        std::string abbrev = getHistoryAbbrev( label );
        unsigned int flag = getHistoryFlag(label);
        printf("%3d : label [%s] abbrev [%s] flag [%u] \n", i, label.c_str(), abbrev.c_str(), flag ); 
    }
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

    m_flags = new Index("GFlagIndex");
    for(unsigned int i=0 ; i < ups.size() ; i++)
    {
        upair_t p = ups[i];
        unsigned int mask = p.first ; 
        unsigned int bitpos = ffs(mask);  // first set bit, 1-based bit position
        unsigned int xmask =  1 << (bitpos-1) ;
        assert( mask == xmask );
        m_flags->add( p.second.c_str(), bitpos ); 
    }



    // TODO: eliminate below by adopting the Index

    m_flag_labels.clear();
    m_flag_codes.clear();
    for(unsigned int i=0 ; i < ups.size() ; i++)
    {   
        upair_t p = ups[i];
        m_flag_codes.push_back(p.first);         
        m_flag_labels.push_back(p.second + m_tail); // include tail for readability of lists and consistency          
    }   
    m_flag_selection = initBooleanSelection(m_flag_labels.size());

    makeFlagAbbrev();
}





void Types::makeFlagAbbrev()
{
    typedef std::map<std::string,std::string> MSS ; 
    MSS s ; 
    s["CERENKOV"]          = "CK" ;
    s["SCINTILLATION"]     = "SC" ;
    s["MISS"]              = "MI" ;
    s["BULK_ABSORB"]       = "AB" ;
    s["BULK_REEMIT"]       = "RE" ;
    s["BULK_SCATTER"]      = "BS" ;
    s["SURFACE_DETECT"]    = "SD" ;
    s["SURFACE_ABSORB"]    = "SA" ;
    s["SURFACE_DREFLECT"]  = "DR" ;
    s["SURFACE_SREFLECT"]  = "SR" ;
    s["BOUNDARY_REFLECT"]  = "BR" ;  
    s["BOUNDARY_TRANSMIT"] = "BT" ;
    s["NAN_ABORT"]         = "NA" ;

    for(MSS::iterator it=s.begin() ; it!=s.end() ; it++)
    {
        std::string label = it->first + m_tail ; 
        std::string abbrev = it->second + m_tail ; 

        m_flag2abbrev[label] = abbrev ; 
        m_abbrev2flag[abbrev] = label ; 

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
         std::string check = getHistoryString(code);
         if(strcmp(label.c_str(),check.c_str()) != 0 )
         {
             printf("mismatch\n");
         }
         std::string abbrev = getHistoryAbbrev(label) ;
         printf(" %10d : %10x : [%s] : [%s] :  %2s :  %d \n", code, code, label.c_str(), check.c_str(), abbrev.c_str(), m_flag_selection[i] );
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



std::string Types::getSequenceString(unsigned long long seq)
{
     // cf env/numerics/thrustrap/Flags.cc

    unsigned int bits = sizeof(seq)*8 ; 
    unsigned int slots = bits/4 ; 

    std::stringstream ss ; 
    for(unsigned int i=0 ; i < slots ; i++)
    {
        unsigned int i4 = i*4 ; 
        unsigned long long mask = (0xFull << i4) ;   // tis essential to use 0xFull rather than 0xF for going beyond 32 bit
        unsigned long long portion = (seq & mask) >> i4 ;  
        unsigned int code = portion ;
        std::string name = getHistoryString( 0x1 << (code-1) );
     /*
        std::cout << std::setw(3) << std::dec << i 
                  << std::setw(30) << std::hex << portion 
                  << std::setw(30) << std::hex << code
                  << std::setw(30) << name   
                  << std::endl ; 
      */

        ss << name  ; 
    }
    return ss.str() ; 
}






unsigned long long Types::convertSequenceString(std::string& seq, Item_t etype, bool hex)
{
    std::string lseq(seq);
    unsigned int elen(0);
    unsigned int nelem(0);
    prepSequenceString(lseq, elen, nelem, hex);

    unsigned long long bseq = 0ull ; 

    for(unsigned int i=0 ; i < nelem ; i++)
    {
        std::string sub = seq.substr(i*elen, elen) ;
        unsigned int bitpos = getAbbrevInvertAsCode(sub, etype, hex);
        //assert(bitpos < 16);
        if(bitpos > 15) LOG(warning) << "Types::convertSequenceString bitpos too big " << bitpos ;  

        unsigned long long ull = bitpos ; 
        unsigned long long msk = ull << (i*4) ; 
        //assert(i*4 < sizeof(unsigned long long)*8 );
        if(!(i*4 < sizeof(unsigned long long)*8 ))
        {
            LOG(warning) << "Types::convertSequenceString too many bits "
                         << " i4 " << i*4 
                         << " seq " << seq 
                         ;
        }
        bseq |= msk ; 
    }  
    return bseq ; 
}



std::string Types::decodeSequenceString(std::string& seq, Item_t etype, bool hex)
{
    std::string lseq(seq);
    unsigned int elen(0);
    unsigned int nelem(0);
    prepSequenceString(lseq, elen, nelem, hex);

    std::stringstream ss ;
    for(unsigned int i=0 ; i < nelem ; i++)
    {
        std::string sub = lseq.substr(i*elen, elen) ;
        std::string label = getAbbrevInvert(sub, etype, hex );
        ss  << label ; // no spacing needed, the tail spacer is internal
    }  
    return ss.str();
}


std::string Types::abbreviateHexSequenceString(std::string& seq, Item_t etype)
{
    bool hex = true ; 

    std::string lseq(seq);
    unsigned int elen(0);
    unsigned int nelem(0);
    prepSequenceString(lseq, elen, nelem, hex);

    std::stringstream ss ;
    for(unsigned int i=0 ; i < nelem ; i++)
    {
        std::string sub = lseq.substr(i*elen, elen) ;
        std::string label = getAbbrevInvert(sub, etype, hex );
        std::string abbr  = getAbbrev(label, etype);
        // not a noop when hex

        ss  << abbr ; // no spacing needed, the tail spacer is internal
    }  
    return ss.str();
}



void Types::prepSequenceString(std::string& lseq, unsigned int& elen, unsigned int& nelem, bool hex)
{
   /*
      Non-hex sequence strings look like this with 2+1 chars per element

            [CE BT KR BT BT BT BT BT BT BT ]

      hex ones have one char per element  and are reversed
            [cccccc3c1]

   */

    if(hex)
    {
        elen = 1 ; 
        nelem = lseq.size();
        std::reverse(lseq.begin(), lseq.end());
    }
    else
    {
        const char* tail = getTail(); 
        elen = 2 + strlen(tail);
        if(lseq.size() % elen != 0)
        {
            LOG(fatal)<<"Types::prepSequenceString "
                      << " lseq " << lseq 
                      << " elen " << elen
                      << " tail [" << tail << "]" ;  
        }
        assert(lseq.size() % elen == 0);
        nelem = lseq.size()/elen ;
    }
}

