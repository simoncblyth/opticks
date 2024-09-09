#pragma once
/**
SMeshGroup.h : collection of SMesh subs and names
===================================================

Persists as folder with int keys. 


**/
struct SMesh ; 
#include "SBitSet.h"


struct SMeshGroup
{
    std::vector<const SMesh*> subs ;
    std::vector<std::string> names ;

    SMeshGroup(); 

    static SMeshGroup* MakeCopy(const SMeshGroup* src, const SBitSet* elv ); 
    SMeshGroup* copy(const SBitSet* elv=nullptr) const ; 


    NPFold* serialize() const ; 
    void save(const char* dir) const ; 
    static SMeshGroup* Import(const NPFold* fold ); 
    void import(const NPFold* fold ); 

    std::string descRange() const ; 
};

inline SMeshGroup::SMeshGroup(){} 



/**
SMeshGroup::MakeCopy
---------------------

NB if none of the subs from the src SMeshGroup are selected by elv a nullptr is returned

**/


inline SMeshGroup* SMeshGroup::MakeCopy(const SMeshGroup* src, const SBitSet* elv ) // static
{
    size_t s_num_subs = src->subs.size(); 
    size_t s_num_names = src->names.size(); 
    assert( s_num_subs == s_num_names ); 

    SMeshGroup* dst = nullptr ; 
    for(size_t i=0 ; i < s_num_subs ; i++ )
    {   
        const SMesh* s_sub = src->subs[i] ; 
        const std::string& s_name = src->names[i] ; 
        int s_lvid = s_sub->lvid ; 

        bool selected = elv == nullptr ? true : elv->is_set(s_lvid) ;  
        if(!selected) continue ; 

        const SMesh* d_sub = s_sub->copy(); 
        std::string d_name = s_name ;   

        if(!dst) dst = new SMeshGroup ; 
        dst->subs.push_back(d_sub) ;  
        dst->names.push_back(d_name) ; 
    }
    return dst ; 
} 

inline SMeshGroup* SMeshGroup::copy(const SBitSet* elv) const
{
    return MakeCopy(this, elv); 
} 










inline NPFold* SMeshGroup::serialize() const 
{
    NPFold* fold = new NPFold ; 
    int num_sub = subs.size(); 
    for(int i=0 ; i < num_sub ; i++)
    {
        const SMesh* sub = subs[i]; 
        const char* name = SMesh::FormName(i) ; 
        fold->add_subfold( name, sub->serialize() ); 
    }
    fold->names = names ;
    return fold ; 
}

inline void SMeshGroup::save(const char* dir) const 
{
    NPFold* fold = serialize(); 
    fold->save(dir); 
}

inline SMeshGroup* SMeshGroup::Import(const NPFold* fold )
{
    SMeshGroup* mg = new SMeshGroup ; 
    mg->import(fold); 
    return mg ; 
}

inline void SMeshGroup::import(const NPFold* fold )
{
    int num_sub = fold->get_num_subfold() ;
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = fold->get_subfold(i); 
        const SMesh* m = SMesh::Import(sub) ;  
        subs.push_back(m); 
    }
}

inline std::string SMeshGroup::descRange() const 
{
    int num_subs = subs.size(); 
    std::stringstream ss ; 
    ss << "[SMeshGroup::descRange num_subs" << num_subs << "\n" ; 
    for(int i=0 ; i < num_subs ; i++) ss << subs[i]->descRange() << "\n" ; 
    ss << "]SMeshGroup::descRange\n" ; 
    std::string str = ss.str() ;
    return str ;  
}

 
