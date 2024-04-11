#pragma once
/**
SMeshGroup.h : collection of subs SMesh and names
===================================================

Persists as folder with int keys. 


**/
struct SMesh ; 

struct SMeshGroup
{
    std::vector<const SMesh*> subs ;
    std::vector<std::string> names ;

    SMeshGroup(); 

    NPFold* serialize() const ; 
    void save(const char* dir) const ; 
    static SMeshGroup* Import(const NPFold* fold ); 
    void import(const NPFold* fold ); 

    std::string descRange() const ; 
};

inline SMeshGroup::SMeshGroup(){} 

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

 
