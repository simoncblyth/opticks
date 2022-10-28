#pragma once
/**
SProp.h : Loads text property files into arrays 
================================================

The persisting is handled by a contained NPFold 
but the loading needs to follow SProp specific conventions 
and also use NP::ArrayFromTxtFile 

Of course after initial loading from the TxtFiles 
the NPFold can be saved and loaded and combined with other
NPFold just like standard ones. 
 
**/

#include "NPFold.h"


struct SProp
{
    static constexpr const char* kNP_PROP_BASE = "NP_PROP_BASE"  ; 
    static constexpr const bool VERBOSE = false ; 

    static const char* Resolve(const char* spec_or_path); 
    static SProp* Load(const char* spec_or_path); 

    static int Compare(const FTSENT** one, const FTSENT** two);
    int  load_fts(const char* base) ;
    void load_array(const char* base, const char* name) ;      

    SProp(const char* spec, const char* symbol=nullptr);
    void init(); 
    std::string desc() const ; 

    const char* spec ;  
    const char* symbol ; 
    const char* base ;  
    NPFold* fold ; 
};

/**
SProp::Resolve
-----------------

spec is expected to be of the following form::

   Material.Pyrex
   Material.Vacuum
   PMTProperty.R12860
   PMTProperty.NNVTMCP
   PMTProperty.NNVTMCP_HiQE
 
Which corresponds to a file system directory 
beneath $NP_PROP_FOLD that contains property txt 
files that follow the convention of not having 
and "." in their names. 

This spec is converted into directory paths such as::

   $NP_PROP_BASE/Material/Pyrex
   $NP_PROP_BASE/Material/Vacuum 
   $NP_PROP_BASE/PMTProperty/R12860
   $NP_PROP_BASE/PMTProperty/NNVTMCP
   $NP_PROP_BASE/PMTProperty/NNVTMCP_HiQE

**/

inline const char* SProp::Resolve(const char* spec) // static 
{
    unsigned num_dot = NP::CountChar(spec, '.') ; 
    const char* relp = num_dot > 0 ? NP::ReplaceChar(spec, '.', '/' ) : spec ; 
    const char* base_ = getenv(kNP_PROP_BASE) ; 
    const char* base  = base_ ? base_ : "/tmp" ; 

    std::stringstream ss ; 
    ss << base << "/" << relp  ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

inline int SProp::Compare(const FTSENT** one, const FTSENT** two)
{
    return (strcmp((*one)->fts_name, (*two)->fts_name));
}

inline int SProp::load_fts(const char* base_) 
{
    char* base = const_cast<char*>(base_);  
    char* basepath[2] {base, nullptr};

    FTS* fs = fts_open(basepath,FTS_COMFOLLOW|FTS_NOCHDIR,&Compare);
    if(fs == nullptr) return 1 ; 

    FTSENT* node = nullptr ;
    while((node = fts_read(fs)) != nullptr)
    {   
        switch (node->fts_info) 
        {   
            case FTS_D :
                break;
            case FTS_F :
            case FTS_SL:
                {   
                    char* name= node->fts_path+strlen(base)+1 ;
                    load_array(base, name) ; 
                }   
                break;
            default:
                break;
        }   
    }   
    fts_close(fs);
    return 0 ; 
}

/**
SProp::load_array
-------------------

This follows the somewhat unusual convention that 
the property txt files have no "." in their names.

Files with "." in their names are ignored. 

**/

inline void SProp::load_array(const char* base, const char* name)
{
    unsigned name_dot = NP::CountChar(name,'.'); 

    if(VERBOSE) std::cout 
        << "SProp::load_array"
        << " base " << base
        << " name " << name
        << " name_dot " << name_dot
        << std::endl
        ;
    if(name_dot > 0) return ; 

    NP* a = NP::ArrayFromTxtFile<double>(base, name) ; 

    if(VERBOSE) std::cout 
        << " name " << std::setw(30) << name 
        << " a " << ( a ? a->sstr() : "-" )
        << std::endl
        ;

    fold->add(name, a );  // NB the NPFold keys will have .npy added
}

inline SProp::SProp(const char* spec_, const char* symbol_)
    :
    spec(spec_ ? strdup(spec_) : nullptr),
    symbol( symbol_ ? strdup(symbol_) : nullptr ), 
    base(spec  ? Resolve(spec) : nullptr),  
    fold(new NPFold)
{
    init(); 
}

inline void SProp::init()
{
    load_fts(base); 
}

inline std::string SProp::desc() const 
{
    std::stringstream ss ; 
    ss << "SProp " 
       << ( symbol ? symbol : "-" )
       << spec 
       << base << std::endl 
       << fold->desc()
       ;
    std::string s = ss.str() ; 
    return s ; 
}

