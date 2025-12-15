#pragma once
/**
SMeshGroup.h : collection of SMesh subs and names
===================================================

Persists as folder with int keys.


**/
struct SMesh ;
#include "SBitSet.h"
#include <string>
#include <vector>


struct SMeshGroup
{
    static constexpr const bool DUMP = false ;
    std::vector<const SMesh*> subs ;
    std::vector<std::string> names ;

    SMeshGroup();

    static SMeshGroup* MakeCopy(const SMeshGroup* src, const SBitSet* elv );
    SMeshGroup* copy(const SBitSet* elv=nullptr) const ;


    NPFold* serialize() const ;
    void save(const char* dir) const ;
    static SMeshGroup* Import(const NPFold* fold );
    void import(const NPFold* fold );

    std::string descRange(const std::vector<std::string>* soname=nullptr) const ;
    static std::string DescSubMesh(const std::vector<const SMesh*>& subs, const std::vector<std::string>* soname);
    void findSubMesh(std::vector<const SMesh*>& subs, int lvid) const;

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
        int s_sub_lvid = s_sub->lvid ;
        bool selected = ( elv == nullptr || s_sub_lvid == -1 ) ? true : elv->is_set(s_sub_lvid) ;

        if(DUMP) std::cout
            << "SMeshGroup::MakeCopy"
            << " i " << i
            << " s_name " << s_name
            << " s_sub_lvid " << s_sub_lvid
            << " s_num_subs " << s_num_subs
            << " s_num_names " << s_num_names
            << "\n"
            ;

        //assert( s_sub_lvid > -1 );
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
    if(DUMP) std::cout << "[SMeshGroup::Import \n" ;
    SMeshGroup* mg = new SMeshGroup ;
    mg->import(fold);
    if(DUMP) std::cout << "]SMeshGroup::Import \n" ;
    return mg ;
}

inline void SMeshGroup::import(const NPFold* fold )
{
    int num_sub = fold->get_num_subfold() ;
    if(DUMP) std::cout << "[SMeshGroup::import num_sub " << num_sub << "\n" ;
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = fold->get_subfold(i);
        if(DUMP) std::cout << ".SMeshGroup::import sub.desc " << sub->desc() << "\n" ;

        const SMesh* m = SMesh::Import(sub) ;
        if(DUMP) std::cout << ".SMeshGroup::import SMesh::Import(sub).lvid" << m->lvid << "\n" ;
        subs.push_back(m);
    }
    names = fold->names ;
    if(DUMP) std::cout << "]SMeshGroup::import num_sub " << num_sub << "\n" ;
}

inline std::string SMeshGroup::descRange(const std::vector<std::string>* soname) const
{
    return DescSubMesh(subs, soname);
}

inline std::string SMeshGroup::DescSubMesh(const std::vector<const SMesh*>& subs, const std::vector<std::string>* soname)
{
    size_t num_so = soname ? soname->size() : 0 ;
    size_t num_subs = subs.size();
    std::stringstream ss ;
    ss << "[SMeshGroup::DescSubMesh num_subs " << num_subs << "\n" ;
    for(size_t i=0 ; i < subs.size() ; i++)
    {
        const SMesh* sub = subs[i];
        const char* so = sub->lvid >= 0 && sub->lvid < int(num_so) ? (*soname)[sub->lvid].c_str() : nullptr ;
        ss
           << std::setw(4) << i
           << " : " << sub->descRange()
           << " so[" << ( so ? so : "-" ) << "]\n"
           ;
    }
    ss << "]SMeshGroup::DescSubMesh\n" ;
    std::string str = ss.str() ;
    return str ;
}


inline void SMeshGroup::findSubMesh(std::vector<const SMesh*>& collect_subs, int lvid) const
{
    size_t num_subs = subs.size();
    for(size_t i=0 ; i < num_subs ; i++)
    {
        const SMesh* sub = subs[i];
        if(lvid == sub->lvid) collect_subs.push_back(sub); 
    }
}










