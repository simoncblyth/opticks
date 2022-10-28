#pragma once
/**
SProp.h : Loads text property files into arrays 
================================================

The persisting is handled by a contained NPFold 
but the loading needs to follow SProp specific conventions 
and also use NP::ArrayFromTxtFile 

Of course after initial loading from the TxtFile 
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
    void load_array(const char* base, const char* relp) ;      

    SProp(); 
    NPFold* f ; 
};

inline const char* SProp::Resolve(const char* spec_or_path) // static 
{
    unsigned num_dot = NP::CountChar(spec_or_path, '.') ; 
    const char* relp = num_dot > 1 ? NP::ReplaceChar(spec_or_path, '.', '/' ) : spec_or_path ; 
    const char* base = getenv(kNP_PROP_BASE) ; 
    std::stringstream ss ; 
    ss << ( base ? base : "/tmp" ) << "/" << relp  ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}

inline SProp* SProp::Load(const char* spec_or_path) // static
{
    const char* base = Resolve(spec_or_path); 
    if(VERBOSE) std::cout 
        << "SProp::Load"
        << " spec_or_path " << spec_or_path
        << " base " << ( base ? base : "-" )
        << std::endl 
        ;

    SProp* p = new SProp ; 
    p->load_fts(base); 
    return p ; 
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
                    char* relp = node->fts_path+strlen(base)+1 ;
                    load_array(base, relp) ; 
                }   
                break;
            default:
                break;
        }   
    }   
    fts_close(fs);
    return 0 ; 
}

inline void SProp::load_array(const char* base, const char* relp)
{
    unsigned relp_dot = NP::CountChar(relp,'.'); 

    if(VERBOSE) std::cout 
        << "SProp::load_array"
        << " base " << base
        << " relp " << relp
        << " relp_dot " << relp_dot
        << std::endl
        ;
    if(relp_dot > 0) return ; 

    NP* a = NP::ArrayFromTxtFile<double>(base, relp) ; 

    std::cout 
        << " relp " << std::setw(30) << relp 
        << " a " << ( a ? a->sstr() : "-" )
        << std::endl
        ;
}

inline SProp::SProp()
   :
   f(new NPFold)
{
}

