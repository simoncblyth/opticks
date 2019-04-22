#include "BFile.hh"
#include "BParameters.hh"
#include "BStr.hh"

#include "NPYMeta.hpp"

#include "PLOG.hh"

const char* NPYMeta::META = "meta.json" ;
const char* NPYMeta::ITEM_META = "item_meta.json" ;

std::string NPYMeta::MetaPath(const char* dir, int idx)  // static 
{
    std::string path = idx == -1 ? BFile::FormPath(dir, META) : BFile::FormPath(dir, BStr::itoa(idx), ITEM_META) ;
    return path ; 
}
bool NPYMeta::ExistsMeta(const char* dir, int idx)  // static
{
    std::string path = MetaPath(dir, idx) ;
    return BFile::ExistsFile(path.c_str()) ;
}
BParameters* NPYMeta::LoadMetadata(const char* dir, int idx ) // static
{
    std::string path = MetaPath(dir, idx) ;
    return BParameters::Load(path.c_str()) ; 
}


NPYMeta::NPYMeta()
{
}

BParameters* NPYMeta::getMeta(int idx) const
{
    return m_meta.count(idx) == 1 ? m_meta.at(idx) : NULL ; 
}
bool NPYMeta::hasMeta(int idx) const
{
    return getMeta(idx) != NULL ; 
}

template<typename T>
T NPYMeta::getValue(const char* key, const char* fallback, int item) const 
{
    BParameters* meta = getMeta(item);  
    return meta ? meta->get<T>(key, fallback) : BStr::LexicalCast<T>(fallback) ;
}

template<typename T>
void NPYMeta::setValue(const char* key, T value, int item)
{
    if(!hasMeta(item)) m_meta[item] = new BParameters ; 
    BParameters* meta = getMeta(item);  
    assert( meta ) ; 
    return meta->set<T>(key, value) ;
}

void NPYMeta::load(const char* dir, int num_item) 
{
    for(int item=-1 ; item < num_item ; item++)
    {
        BParameters* meta = LoadMetadata(dir, item);
        if(meta) m_meta[item] = meta ; 
    } 
}
void NPYMeta::save(const char* dir) const 
{
    typedef std::map<int, BParameters*> MIP ; 
    for(MIP::const_iterator it=m_meta.begin() ; it != m_meta.end() ; it++)
    {
        int item = it->first ; 
        BParameters* meta = it->second ; 
        std::string metapath = MetaPath(dir, item) ;
        assert(meta); 
        meta->save(metapath.c_str()); 
    }    
}


template NPY_API void NPYMeta::setValue<double>(const char*, double, int);
template NPY_API void NPYMeta::setValue<float>(const char*, float, int);
template NPY_API void NPYMeta::setValue<int>(const char*, int, int);
template NPY_API void NPYMeta::setValue<bool>(const char*, bool, int);
template NPY_API void NPYMeta::setValue<std::string>(const char*, std::string, int);

template NPY_API std::string NPYMeta::getValue<std::string>(const char*, const char*, int) const ;
template NPY_API int         NPYMeta::getValue<int>(const char*, const char*, int) const ;
template NPY_API double      NPYMeta::getValue<double>(const char*, const char*, int) const ;
template NPY_API float       NPYMeta::getValue<float>(const char*, const char*, int) const ;
template NPY_API bool        NPYMeta::getValue<bool>(const char*, const char*, int) const ;


