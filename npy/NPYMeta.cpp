#include "BFile.hh"

#ifdef OLD_PARAMETERS
#include "X_BParameters.hh"
#else
#include "NMeta.hpp"
#endif


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

#ifdef OLD_PARAMETERS
X_BParameters* NPYMeta::LoadMetadata(const char* dir, int idx ) // static
{
    std::string path = MetaPath(dir, idx) ;
    return X_BParameters::Load(path.c_str()) ; 
}
#else
NMeta* NPYMeta::LoadMetadata(const char* dir, int idx ) // static
{
    std::string path = MetaPath(dir, idx) ;
    return NMeta::Load(path.c_str()) ; 
}
#endif


NPYMeta::NPYMeta()
{
}

#ifdef OLD_PARAMETERS
X_BParameters* NPYMeta::getMeta(int idx) const
#else
NMeta* NPYMeta::getMeta(int idx) const
#endif
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
#ifdef OLD_PARAMETERS
    X_BParameters* meta = getMeta(item);  
#else
    NMeta* meta = getMeta(item);  
#endif
    return meta ? meta->get<T>(key, fallback) : BStr::LexicalCast<T>(fallback) ;
}


int NPYMeta::getIntFromString(const char* key, const char* fallback, int item) const
{
#ifdef OLD_PARAMETERS
    X_BParameters* meta = getMeta(item);  
#else
    NMeta* meta = getMeta(item);  
#endif
    return meta ? meta->getIntFromString(key,fallback) : BStr::LexicalCast<int>(fallback) ;
}



template<typename T>
void NPYMeta::setValue(const char* key, T value, int item)
{
#ifdef OLD_PARAMETERS
    if(!hasMeta(item)) m_meta[item] = new X_BParameters ; 
    X_BParameters* meta = getMeta(item);  
#else
    if(!hasMeta(item)) m_meta[item] = new NMeta ; 
    NMeta* meta = getMeta(item);  
#endif

    assert( meta ) ; 
    return meta->set<T>(key, value) ;
}

void NPYMeta::load(const char* dir, int num_item) 
{
    for(int item=-1 ; item < num_item ; item++)
    {
#ifdef OLD_PARAMETERS
        X_BParameters* meta = LoadMetadata(dir, item);
#else
        NMeta* meta = LoadMetadata(dir, item);
#endif
        if(meta) m_meta[item] = meta ; 
    } 
}
void NPYMeta::save(const char* dir) const 
{
#ifdef OLD_PARAMETERS
    typedef std::map<int, X_BParameters*> MIP ; 
#else
    typedef std::map<int, NMeta*> MIP ; 
#endif
    for(MIP::const_iterator it=m_meta.begin() ; it != m_meta.end() ; it++)
    {
        int item = it->first ; 
#ifdef OLD_PARAMETERS
        X_BParameters* meta = it->second ; 
#else
        NMeta* meta = it->second ; 
#endif
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


