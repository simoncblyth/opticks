
#include <map>
#include <vector>
#include <string>

#include "BStr.hh"
#include "BBit.hh"
#include "BRegex.hh"

#include "NMeta.hpp"

#include "PLOG.hh"

#include "Index.hpp"

#include "OpticksFlags.hh"


//const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/graphics/optixrap/cu/photon.h" ;
//const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/opticks/OpticksPhoton.h" ;
//const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/optickscore/OpticksPhoton.h" ;

const char* OpticksFlags::ABBREV_META_NAME = "OpticksFlagsAbbrevMeta.json" ;


const char* OpticksFlags::ENUM_HEADER_PATH = "$OPTICKS_INSTALL_PREFIX/include/OpticksCore/OpticksPhoton.h" ;
//  envvar OPTICKS_INSTALL_PREFIX is set internally by OpticksResource based on cmake config 


const char* OpticksFlags::ZERO_              = "." ;
const char* OpticksFlags::NATURAL_           = "NATURAL" ;
const char* OpticksFlags::FABRICATED_        = "FABRICATED" ;
const char* OpticksFlags::MACHINERY_         = "MACHINERY" ;
const char* OpticksFlags::EMITSOURCE_        = "EMITSOURCE" ;
const char* OpticksFlags::PRIMARYSOURCE_     = "PRIMARYSOURCE" ;
const char* OpticksFlags::GENSTEPSOURCE_     = "GENSTEPSOURCE" ;

const char* OpticksFlags::CERENKOV_          = "CERENKOV" ;
const char* OpticksFlags::SCINTILLATION_     = "SCINTILLATION" ;
const char* OpticksFlags::MISS_              = "MISS" ;
const char* OpticksFlags::OTHER_             = "OTHER" ;
const char* OpticksFlags::BULK_ABSORB_       = "BULK_ABSORB" ;
const char* OpticksFlags::BULK_REEMIT_       = "BULK_REEMIT" ;
const char* OpticksFlags::BULK_SCATTER_      = "BULK_SCATTER" ; 
const char* OpticksFlags::SURFACE_DETECT_    = "SURFACE_DETECT" ;
const char* OpticksFlags::SURFACE_ABSORB_    = "SURFACE_ABSORB" ; 
const char* OpticksFlags::SURFACE_DREFLECT_  = "SURFACE_DREFLECT" ; 
const char* OpticksFlags::SURFACE_SREFLECT_  = "SURFACE_SREFLECT" ; 
const char* OpticksFlags::BOUNDARY_REFLECT_  = "BOUNDARY_REFLECT" ; 
const char* OpticksFlags::BOUNDARY_TRANSMIT_ = "BOUNDARY_TRANSMIT" ; 
const char* OpticksFlags::TORCH_             = "TORCH" ; 
const char* OpticksFlags::G4GUN_             = "G4GUN" ; 
const char* OpticksFlags::NAN_ABORT_         = "NAN_ABORT" ; 
const char* OpticksFlags::BAD_FLAG_          = "BAD_FLAG" ; 

// NB this is duplicating abbrev from /usr/local/opticks/opticksdata/resource/GFlags/abbrev.json
//    TODO: get rid of that 
//
//     as these are so fixed they deserve static enshrinement for easy access from everywhere
//
const char* OpticksFlags::_ZERO              = "  " ;
const char* OpticksFlags::_NATURAL           = "NL" ;
const char* OpticksFlags::_FABRICATED        = "FD" ;
const char* OpticksFlags::_MACHINERY         = "MY" ;
const char* OpticksFlags::_EMITSOURCE        = "SO" ;
const char* OpticksFlags::_PRIMARYSOURCE     = "PS" ;
const char* OpticksFlags::_GENSTEPSOURCE     = "GS" ;

const char* OpticksFlags::_CERENKOV          = "CK" ;
const char* OpticksFlags::_SCINTILLATION     = "SI" ;
const char* OpticksFlags::_TORCH             = "TO" ; 
const char* OpticksFlags::_MISS              = "MI" ;
const char* OpticksFlags::_BULK_ABSORB       = "AB" ;
const char* OpticksFlags::_BULK_REEMIT       = "RE" ;
const char* OpticksFlags::_BULK_SCATTER      = "SC" ; 
const char* OpticksFlags::_SURFACE_DETECT    = "SD" ;
const char* OpticksFlags::_SURFACE_ABSORB    = "SA" ; 
const char* OpticksFlags::_SURFACE_DREFLECT  = "DR" ; 
const char* OpticksFlags::_SURFACE_SREFLECT  = "SR" ; 
const char* OpticksFlags::_BOUNDARY_REFLECT  = "BR" ; 
const char* OpticksFlags::_BOUNDARY_TRANSMIT = "BT" ; 
const char* OpticksFlags::_NAN_ABORT         = "NA" ; 
const char* OpticksFlags::_G4GUN             = "GN" ; 
const char* OpticksFlags::_BAD_FLAG          = "XX" ; 


NMeta* OpticksFlags::makeAbbrevMeta()
{
    NMeta* m = new NMeta ; 
    m->set<std::string>(CERENKOV_ , _CERENKOV); 
    m->set<std::string>(SCINTILLATION_ , _SCINTILLATION); 
    m->set<std::string>(TORCH_ , _TORCH); 
    m->set<std::string>(MISS_ , _MISS); 
    m->set<std::string>(BULK_ABSORB_ , _BULK_ABSORB); 
    m->set<std::string>(BULK_REEMIT_ , _BULK_REEMIT); 
    m->set<std::string>(BULK_SCATTER_ , _BULK_SCATTER); 
    m->set<std::string>(SURFACE_DETECT_ , _SURFACE_DETECT); 
    m->set<std::string>(SURFACE_ABSORB_ , _SURFACE_ABSORB); 
    m->set<std::string>(SURFACE_DREFLECT_ , _SURFACE_DREFLECT); 
    m->set<std::string>(SURFACE_SREFLECT_ , _SURFACE_SREFLECT); 
    m->set<std::string>(BOUNDARY_REFLECT_ , _BOUNDARY_REFLECT); 
    m->set<std::string>(BOUNDARY_TRANSMIT_ , _BOUNDARY_TRANSMIT); 
    m->set<std::string>(NAN_ABORT_ , _NAN_ABORT); 
    return m ; 
}


const char* OpticksFlags::natural_           = "natural" ;
const char* OpticksFlags::fabricated_        = "fabricated" ;
const char* OpticksFlags::machinery_         = "machinery" ;
const char* OpticksFlags::cerenkov_          = "cerenkov" ;
const char* OpticksFlags::scintillation_     = "scintillation" ;
const char* OpticksFlags::torch_             = "torch" ; 
const char* OpticksFlags::emitsource_        = "emitsource" ; 
const char* OpticksFlags::primarysource_     = "primarysource" ; 
const char* OpticksFlags::genstepsource_     = "genstepsource" ; 
const char* OpticksFlags::g4gun_             = "g4gun" ; 
const char* OpticksFlags::other_             = "other" ;


const char* OpticksFlags::Flag(const unsigned int flag)
{
    const char* s = 0 ; 
    switch(flag)
    {
        case 0:                s=ZERO_;break;
        case CERENKOV:         s=CERENKOV_;break;
        case SCINTILLATION:    s=SCINTILLATION_ ;break; 
        case MISS:             s=MISS_ ;break; 
        case BULK_ABSORB:      s=BULK_ABSORB_ ;break; 
        case BULK_REEMIT:      s=BULK_REEMIT_ ;break; 
        case BULK_SCATTER:     s=BULK_SCATTER_ ;break; 
        case SURFACE_DETECT:   s=SURFACE_DETECT_ ;break; 
        case SURFACE_ABSORB:   s=SURFACE_ABSORB_ ;break; 
        case SURFACE_DREFLECT: s=SURFACE_DREFLECT_ ;break; 
        case SURFACE_SREFLECT: s=SURFACE_SREFLECT_ ;break; 
        case BOUNDARY_REFLECT: s=BOUNDARY_REFLECT_ ;break; 
        case BOUNDARY_TRANSMIT:s=BOUNDARY_TRANSMIT_ ;break; 
        case TORCH:            s=TORCH_ ;break; 
        case NAN_ABORT:        s=NAN_ABORT_ ;break; 
        case G4GUN:            s=G4GUN_ ;break; 
        case NATURAL:          s=NATURAL_ ;break; 
        case FABRICATED:       s=FABRICATED_ ;break; 
        case MACHINERY:        s=MACHINERY_;break; 
        case EMITSOURCE:       s=EMITSOURCE_;break; 
        case PRIMARYSOURCE:    s=PRIMARYSOURCE_;break; 
        case GENSTEPSOURCE:    s=GENSTEPSOURCE_;break; 
        default:               s=BAD_FLAG_  ;
                               LOG(debug) << "OpticksFlags::Flag BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec ;             
    }
    return s;
}



const char* OpticksFlags::Abbrev(const unsigned int flag)
{
    const char* s = 0 ; 
    switch(flag)
    {
        case 0:                s=_ZERO;break;
        case CERENKOV:         s=_CERENKOV;break;
        case SCINTILLATION:    s=_SCINTILLATION ;break; 
        case MISS:             s=_MISS ;break; 
        case BULK_ABSORB:      s=_BULK_ABSORB ;break; 
        case BULK_REEMIT:      s=_BULK_REEMIT ;break; 
        case BULK_SCATTER:     s=_BULK_SCATTER ;break; 
        case SURFACE_DETECT:   s=_SURFACE_DETECT ;break; 
        case SURFACE_ABSORB:   s=_SURFACE_ABSORB ;break; 
        case SURFACE_DREFLECT: s=_SURFACE_DREFLECT ;break; 
        case SURFACE_SREFLECT: s=_SURFACE_SREFLECT ;break; 
        case BOUNDARY_REFLECT: s=_BOUNDARY_REFLECT ;break; 
        case BOUNDARY_TRANSMIT:s=_BOUNDARY_TRANSMIT ;break; 
        case TORCH:            s=_TORCH ;break; 
        case NAN_ABORT:        s=_NAN_ABORT ;break; 
        case G4GUN:            s=_G4GUN ;break; 
        case NATURAL:          s=_NATURAL ;break; 
        case FABRICATED:       s=_FABRICATED ;break; 
        case MACHINERY:        s=_MACHINERY;break; 
        case EMITSOURCE:       s=_EMITSOURCE;break; 
        case PRIMARYSOURCE:    s=_PRIMARYSOURCE;break; 
        case GENSTEPSOURCE:    s=_GENSTEPSOURCE;break; 
        default:               s=_BAD_FLAG  ;
                               LOG(verbose) << "OpticksFlags::Abbrev BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec ;             
    }
    return s;
}



unsigned OpticksFlags::EnumFlag(unsigned bitpos)
{
    return bitpos == 0 ? 0 : 0x1 << (bitpos - 1) ;
}

unsigned OpticksFlags::BitPos(unsigned flag)
{
    return BBit::ffs(flag)  ;  
}

unsigned OpticksFlags::AbbrevToFlag( const char* abbrev )
{
    unsigned flag = 0 ; 
    if(!abbrev) return flag ;
 
    for(unsigned f=0 ; f < 32 ; f++) 
    {
        flag = EnumFlag(32-1-f) ; // <-- reverse order so unfound -> 0 
        if(strcmp(Abbrev(flag), abbrev) == 0) break ; 
    }
    return flag ;        
}

unsigned long long OpticksFlags::AbbrevToFlagSequence( const char* abbseq, char delim)
{
   // convert seqhis string eg "TO SR SA" into bigint 0x8ad

   std::vector<std::string> elem ; 
   BStr::split(elem, abbseq,  delim ); 

   unsigned long long seqhis = 0 ; 
   for(unsigned i=0 ; i < elem.size() ; i++)
   {
       unsigned flag = AbbrevToFlag( elem[i].c_str() );
       unsigned bitpos = BitPos(flag) ; 
       unsigned long long shift = i*4 ;  
       seqhis |= ( bitpos << shift )  ; 
   }   
   return seqhis ; 
}



/**

OpticksFlags::AbbrevToFlagValSequence
--------------------------------------

Convert seqmap string into two bigints, 
map values are 1-based, zero signifies None.

=====================  =============  ===========
input seqmap             seqhis         seqval 
=====================  =============  ===========
"TO:0 SR:1 SA:0"          0x8ad          0x121
"TO:0 SC: SR:1 SA:0"      0x8a6d        0x1201
=====================  =============  ===========
 
**/

void OpticksFlags::AbbrevToFlagValSequence( unsigned long long& seqhis, unsigned long long& seqval, const char* seqmap, char edelim)
{
   seqhis = 0ull ; 
   seqval = 0ull ; 

   std::vector<std::pair<std::string, std::string> > ekv ; 
   const char* kvdelim=":" ; 
   BStr::ekv_split( ekv, seqmap, edelim, kvdelim );

   for(unsigned i=0 ; i < ekv.size() ; i++ ) 
   { 
       unsigned long long ishift = i*4 ; 
       std::string skey = ekv[i].first ; 
       std::string sval = ekv[i].second ; 

       unsigned flag = AbbrevToFlag( skey.c_str() );
       unsigned bitpos = BitPos(flag) ; 

       unsigned val1 = sval.empty() ? 0 : 1u + BStr::atoi( sval.c_str() ) ;
 
       seqhis |= ( bitpos << ishift )  ; 
       seqval |= ( val1 << ishift )  ; 

       LOG(debug)
                   << "[" 
                   <<  skey
                   << "] -> [" 
                   <<  sval << "]" 
                   << ( sval.empty() ? "EMPTY" : "" )
                   << " val1 " << val1 
                    ; 
    }

}




unsigned OpticksFlags::PointVal1( const unsigned long long& seqval , unsigned bitpos )
{
    return (seqval >> bitpos*4) & 0xF ; 
}


unsigned OpticksFlags::PointFlag( const unsigned long long& seqhis , unsigned bitpos )
{
    unsigned long long f = (seqhis >> bitpos*4) & 0xF ; 
    unsigned flg = f == 0 ? 0 : 0x1 << (f - 1) ; 
    return flg ; 
}

const char* OpticksFlags::PointAbbrev( const unsigned long long& seqhis , unsigned bitpos )
{
    unsigned flg = PointFlag(seqhis, bitpos );
    return Abbrev(flg);     
}


std::string OpticksFlags::FlagSequence(const unsigned long long seqhis, bool abbrev, int highlight)
{
    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);

    unsigned hi = highlight < 0 ? 16 : highlight ; 

    for(unsigned int i=0 ; i < 16 ; i++)
    {
        unsigned long long f = (seqhis >> i*4) & 0xF ; 
        unsigned int flg = f == 0 ? 0 : 0x1 << (f - 1) ; 
        if(i == hi) ss << "[" ;  
        ss << ( abbrev ? Abbrev(flg) : Flag(flg) ) ;
        if(i == hi) ss << "]" ;  
        ss << " " ; 
    }
    return ss.str();
}


std::string OpticksFlags::FlagMask(const unsigned mskhis, bool abbrev)
{
    std::vector<const char*> labels ; 

    assert( MACHINERY == 0x1 << 17 );
    unsigned lastBit = BBit::ffs(MACHINERY) - 1 ;  
    assert(lastBit == 17 ); 
 
    for(unsigned n=0 ; n <= lastBit ; n++ )
    {
        unsigned flag = 0x1 << n ; 
        if(mskhis & flag) labels.push_back( abbrev ? Abbrev(flag) : Flag(flag) );
    }
    unsigned nlab = labels.size() ; 

    std::stringstream ss ;
    for(unsigned i=0 ; i < nlab ; i++ ) ss << labels[i] << ( i < nlab - 1 ? "|" : ""  ) ; 
    return ss.str();
}


const char* OpticksFlags::SourceType( int code )
{
    const char* name = 0 ; 
    switch(code)
    {
       case NATURAL      :name = NATURAL_       ;break;
       case FABRICATED   :name = FABRICATED_    ;break;
       case MACHINERY    :name = MACHINERY_     ;break;
       case CERENKOV     :name = CERENKOV_      ;break;
       case SCINTILLATION:name = SCINTILLATION_ ;break;
       case TORCH        :name = TORCH_         ;break;
       case G4GUN        :name = G4GUN_         ;break;
       case EMITSOURCE   :name = EMITSOURCE_    ;break;
       case PRIMARYSOURCE:name = PRIMARYSOURCE_ ;break;
       case GENSTEPSOURCE:name = GENSTEPSOURCE_ ;break;
       default           :name = OTHER_         ;break; 
    }
    return name ; 
}

const char* OpticksFlags::SourceTypeLowercase( int code )
{
    const char* name = 0 ; 
    switch(code)
    {
       case NATURAL      :name = natural_       ;break;
       case FABRICATED   :name = fabricated_    ;break;
       case MACHINERY    :name = machinery_     ;break;
       case CERENKOV     :name = cerenkov_      ;break;
       case SCINTILLATION:name = scintillation_ ;break;
       case TORCH        :name = torch_         ;break;
       case G4GUN        :name = g4gun_         ;break;
       case EMITSOURCE   :name = emitsource_    ;break;
       case PRIMARYSOURCE:name = primarysource_ ;break;
       case GENSTEPSOURCE:name = genstepsource_ ;break;
       default           :name = other_         ;break; 
    }
    return name ; 
}

unsigned int OpticksFlags::SourceCode(const char* type)
{
    unsigned int code = 0 ; 
    if(     strcmp(type,natural_)==0)       code = NATURAL ;
    else if(strcmp(type,fabricated_)==0)    code = FABRICATED ;
    else if(strcmp(type,machinery_)==0)     code = MACHINERY ;
    else if(strcmp(type,torch_)==0)         code = TORCH ;
    else if(strcmp(type,cerenkov_)==0)      code = CERENKOV ;
    else if(strcmp(type,scintillation_)==0) code = SCINTILLATION ;
    else if(strcmp(type,g4gun_)==0)         code = G4GUN ;
    else if(strcmp(type,emitsource_)==0)    code = EMITSOURCE ;
    else if(strcmp(type,primarysource_)==0) code = PRIMARYSOURCE ;
    else if(strcmp(type,genstepsource_)==0) code = GENSTEPSOURCE ;
    return code ; 
}




Index* OpticksFlags::getIndex()     const { return m_index ;  } 
NMeta* OpticksFlags::getAbbrevMeta() const { return m_abbrev_meta ;  } 

OpticksFlags::OpticksFlags(const char* path) 
    :
    m_index(parseFlags(path)),
    m_abbrev_meta(makeAbbrevMeta())
{
}


void OpticksFlags::save(const char* installcachedir)
{
    LOG(info) << installcachedir ; 
    m_index->setExt(".ini"); 
    m_index->save(installcachedir);
    m_abbrev_meta->save( installcachedir, ABBREV_META_NAME ); 
}

Index* OpticksFlags::parseFlags(const char* path)
{
    typedef std::pair<unsigned, std::string>  upair_t ;
    typedef std::vector<upair_t>              upairs_t ;
    upairs_t ups ;
    BRegex::enum_regexsearch( ups, path ); 

    const char* reldir = NULL ; 
    Index* index = new Index("GFlags", reldir);
    for(unsigned i=0 ; i < ups.size() ; i++)
    {
        upair_t p = ups[i];
        unsigned mask = p.first ;
        unsigned bitpos = BBit::ffs(mask);  // first set bit, 1-based bit position
        unsigned xmask = 1 << (bitpos-1) ; 
        assert( mask == xmask);

        const char* key = p.second.c_str() ;

        LOG(debug) << " key " << std::setw(20) << key 
                   << " bitpos " << bitpos 
                   ;

        index->add( key, bitpos );
   
        //  OpticksFlagsTest --OKCORE debug 
    }

    unsigned int num_flags = index->getNumItems() ;
    if(num_flags == 0)
    { 
        LOG(fatal)
             << " path " << path 
             << " num_flags " << num_flags 
             << " " << ( index ? index->description() : "NULL index" )
             ;
    }
    assert(num_flags > 0 && "missing flags header ? : you need to update OpticksFlags::ENUM_HEADER_PATH ");

    return index ; 
}


