/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "BFile.hh"
#include "BStr.hh"

#include "NMeta.hpp"
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

NMeta* NPYMeta::LoadMetadata(const char* dir, int idx ) // static
{
    std::string path = MetaPath(dir, idx) ;
    return NMeta::Load(path.c_str()) ; 
}

NPYMeta::NPYMeta()
{
}

NMeta* NPYMeta::getMeta(int idx) const
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
    NMeta* meta = getMeta(item);  
    return meta ? meta->get<T>(key, fallback) : BStr::LexicalCast<T>(fallback) ;
}

int NPYMeta::getIntFromString(const char* key, const char* fallback, int item) const
{
    NMeta* meta = getMeta(item);  
    return meta ? meta->getIntFromString(key,fallback) : BStr::LexicalCast<int>(fallback) ;
}

template<typename T>
void NPYMeta::setValue(const char* key, T value, int item)
{
    if(!hasMeta(item)) m_meta[item] = new NMeta ; 
    NMeta* meta = getMeta(item);  

    assert( meta ) ; 
    return meta->set<T>(key, value) ;
}

void NPYMeta::load(const char* dir, int num_item) 
{
    for(int item=-1 ; item < num_item ; item++)
    {
        NMeta* meta = LoadMetadata(dir, item);
        if(meta) m_meta[item] = meta ; 
    } 
}
void NPYMeta::save(const char* dir) const 
{
    typedef std::map<int, NMeta*> MIP ; 
    for(MIP::const_iterator it=m_meta.begin() ; it != m_meta.end() ; it++)
    {
        int item = it->first ; 
        NMeta* meta = it->second ; 
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


