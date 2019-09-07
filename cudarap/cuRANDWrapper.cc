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

#include <cstdio>
#include <cassert>

// sysrap-
#include "PLOG.hh"
#include "SDigest.hh"

#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchCommon.hh"
#include "LaunchSequence.hh"

#include "curand_kernel.h"


const plog::Severity cuRANDWrapper::LEVEL = PLOG::EnvLevel("cuRANDWrapper", "DEBUG") ; 


/**
cuRANDWrapper::instanciate
---------------------------

Canonically invoked from ORng::init and from cuRANDWrapperTest::main 

**/

cuRANDWrapper* cuRANDWrapper::instanciate(
    unsigned num_items, 
    const char* cachedir,
    unsigned long long seed,
    unsigned long long offset,
    unsigned max_blocks,
    unsigned threads_per_block,
    bool verbose
)
{
    LOG(LEVEL) 
        << " num_items " << num_items 
        << " cachedir " << ( cachedir ? cachedir : "-" ) 
        ;

    LaunchSequence* seq = new LaunchSequence( num_items, threads_per_block, max_blocks ) ;
    cuRANDWrapper* crw = new cuRANDWrapper(seq, seed, offset, verbose);

    if(cachedir)
    {
        LOG(LEVEL) << " cache enabled  " << cachedir ; 
        crw->setCacheDir(cachedir);
        crw->setCacheEnabled(true);
    }
    else
    {
        LOG(LEVEL) << " cache disabled  " ; 
        crw->setCacheEnabled(false);
    }
    return crw ; 
}


/**
cuRANDWrapper::cuRANDWrapper
------------------------------


**/


cuRANDWrapper::cuRANDWrapper( LaunchSequence* launchseq, unsigned long long seed, unsigned long long offset, bool verbose )
    :
    m_seed(seed),
    m_offset(offset),
    m_verbose(verbose),
    m_dev_rng_states(0),
    m_host_rng_states(0),
    m_test(0),
    m_launchseq(launchseq),
    m_imod(100000),
    m_cache_dir(0),
    m_cache_enabled(true),
    m_owner(false),
    m_first_resize(true)
{
    setCacheDir("/tmp"); 
}



const char* cuRANDWrapper::getCachePath() 
{
    char buf[256];
    snprintf(buf, 256, "%s/cuRANDWrapper_%u_%llu_%llu.bin", 
                 m_cache_dir,
                 getItems(),
                 m_seed,
                 m_offset); 

    return strdup(buf) ; 
}

unsigned cuRANDWrapper::getSeed() const 
{  
    return m_seed ; 
}
unsigned cuRANDWrapper::getOffset() const 
{ 
    return m_offset ; 
}
bool cuRANDWrapper::isVerbose() const 
{
    return m_verbose ; 
}
LaunchSequence* cuRANDWrapper::getLaunchSequence() const 
{ 
    return m_launchseq ; 
}


unsigned cuRANDWrapper::getItems() const 
{ 
    return m_launchseq->getItems() ; 
}

void cuRANDWrapper::setItems(unsigned int items)
{
    m_launchseq->setItems(items); 
}

void cuRANDWrapper::setCacheEnabled(bool enabled)
{ 
    m_cache_enabled = enabled ; 
}
bool cuRANDWrapper::hasCacheEnabled() const 
{
    return m_cache_enabled ; 
}




void cuRANDWrapper::setDevRngStates(CUdeviceptr dev_rng_states, bool owner )
{
    m_dev_rng_states = dev_rng_states ;
    m_owner = owner ; 
} 
CUdeviceptr cuRANDWrapper::getDevRngStates() const  
{ 
    return m_dev_rng_states ; 
}
bool cuRANDWrapper::isOwner() const 
{ 
    return m_owner ; 
} 

curandState* cuRANDWrapper::getHostRngStates() const 
{ 
    return m_host_rng_states ; 
}

void cuRANDWrapper::setImod(unsigned int imod)
{
    m_imod = imod ; 
}





void cuRANDWrapper::setCacheDir(const char* dir)
{
    m_cache_dir = strdup(dir);
}

 


char* cuRANDWrapper::digest()
{
    if(!m_host_rng_states) return 0 ;

    SDigest dig ;
    dig.update( (char*)m_host_rng_states, sizeof(curandState)*getItems()) ; 
    return dig.finalize();
}

char* cuRANDWrapper::testdigest()
{
    SDigest dig ;
    dig.update( (char*)m_test, sizeof(float)*getItems()) ; 
    return dig.finalize();
}

int cuRANDWrapper::Test()
{
    test_rng("test_0");
    test_rng("test_1");
    test_rng("test_2");
    test_rng("test_3");
    test_rng("test_4");

    return 0 ;
}

void cuRANDWrapper::devicesync()
{
    devicesync_wrapper();
}


/**
cuRANDWrapper::test_rng
--------------------------

Generate random floats using the curandState device buffer m_dev_rng_states, 
note that when update_states is false the same sequence of randoms are generated
at each call.  

Opticks generate.cu currently does not do the equivalent of this 
update_states as the entire photons simulation is handled in one OptiX kernel, 
meaning there is no need to persist states. 

**/

void cuRANDWrapper::test_rng(const char* tag, bool update_states )
{
    LaunchSequence* seq = m_launchseq->copy() ;

    seq->setTag(tag);

    unsigned items = seq->getItems();

    m_test  = (float*)malloc(items*sizeof(float));

    test_rng_wrapper( seq, m_dev_rng_states, m_test, update_states );

    char* test_digest = testdigest();

    LOG(info) 
        << " tag " << tag 
        << " items " << items  
        << " imod " << m_imod
        << " test_digest " << test_digest
        ; 

    for(unsigned i=0 ; i<items ; ++i)
    {
        if( i < 10 || i % m_imod == 0 )
            std::cout  << std::fixed << std::setprecision(4) <<  m_test[i] << " "  ; 
    }
    std::cout << std::endl ;  

    free(test_digest);
    free(m_test);

    m_launchrec.push_back(seq); 
}


void cuRANDWrapper::Summary(const char* msg)
{
    unsigned nrec = m_launchrec.size();
    for(unsigned i=0 ; i < nrec ; i++)
    {
        LaunchSequence* seq = m_launchrec[i];  
        seq->Summary(msg); 
    }
}



/**
cuRANDWrapper::Dump
---------------------

By inspection boxmuller_extra boxmuller_extra_double seem un-initialized
with flags unset, from docs appears to only be used for
curand_normal calls 

**/

void cuRANDWrapper::Dump(const char* msg, unsigned int imod)
{
    char* dig = digest() ;
    printf("%s digest %s \n", msg, dig);
    free(dig);

    curandState* rng_states = (curandState*)m_host_rng_states ;

    unsigned int items = getItems();
    for(unsigned int i = 0 ; i < items ; ++i )
    {
        if(i % imod != 0) continue ;   
        curandState& rng = rng_states[i] ;
        printf("i %10u/%10u : d %10u v %10u %10u %10u %10u %10u boxmuller_extra %10.4f _extra_double %10.4f \n", 
            i,
            items,
            rng.d, 
            rng.v[0], 
            rng.v[1], 
            rng.v[2], 
            rng.v[3], 
            rng.v[4],
            rng.boxmuller_flag        ? rng.boxmuller_extra        : -1.f , 
            rng.boxmuller_flag_double ? rng.boxmuller_extra_double : -1.f );

    }
}

/**
cuRANDWrapper::InitFromCacheIfPossible
----------------------------------------

When cache is enabled and exists (the normal case)
this just loads it. When enabled and not existing Init and Save
are invoked doing the CUDA launches and saving to cache. 

**/

int cuRANDWrapper::InitFromCacheIfPossible()
{
    LOG(LEVEL) ; 
    if(!hasCacheEnabled())
    {
        LOG(LEVEL) << " cache disabled -> Init " ; 
        Init();
    }
    else
    {
        if(hasCache())
        {
            LOG(LEVEL) << " has-cache -> Load " ; 
            Load();
        }
        else
        {
            LOG(LEVEL) << " no-cache -> Init+Save " ; 
            Init();
            Save();
        }
    }
    return 0 ;
}




int cuRANDWrapper::hasCache()
{
    const char* path = getCachePath() ;
    int rc = hasCache(path);
    return rc ; 
}



/**
cuRANDWrapper::Allocate
--------------------------

Device buffer m_dev_rng_states

**/

int cuRANDWrapper::Allocate()
{
    LOG(LEVEL) << "["  ; 
    m_owner = true ; 
    m_dev_rng_states = allocate_rng_wrapper(m_launchseq);
    devicesync();  
    LOG(LEVEL) << "]"  ; 
    return 0 ;
}


/**
cuRANDWrapper::Free
---------------------

Device buffer m_dev_rng_states

**/

int cuRANDWrapper::Free()
{
    LOG(LEVEL) ; 
    assert(isOwner());

    free_rng_wrapper(m_dev_rng_states);
    devicesync();  
    return 0 ;
}


/**
cuRANDWrapper::Init
--------------------

Performs multiple CUDA launches to curand_init
the curandState device buffer m_dev_rng_states.

**/

int cuRANDWrapper::Init()
{
    LOG(LEVEL) << "["  ; 

    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag("init");
    init_rng_wrapper( seq, m_dev_rng_states, m_seed, m_offset);
    m_launchrec.push_back(seq); 

    devicesync();  
    LOG(LEVEL) << "]"  ; 
    return 0 ;
}


/**
cuRANDWrapper::Save
--------------------

1. copy curandState buffer from device to host 
2. save to cache path file

**/

int cuRANDWrapper::Save()
{
    LOG(LEVEL) << "["  ; 
    const char* path = getCachePath() ;

    m_host_rng_states = copytohost_rng_wrapper(m_launchseq, m_dev_rng_states);

    devicesync();

    char* save_digest = digest() ;

    LOG(LEVEL)
        << " items " << getItems()
        << " path " << path 
        << " save_digest " << save_digest
        ; 

    int rc = SaveToFile(path);

    free(save_digest);
    LOG(LEVEL) << "]"  ; 
    return rc ; 
}



/**
cuRANDWrapper::Load
---------------------

1. loads RNG states from cache
2. uploads to device
3. when rountrip testing is enabled, immediately copies back to 
   device and asserts that digests match  


**/

int cuRANDWrapper::Load()
{
    LOG(LEVEL) << "["  ; 
    const char* path = getCachePath() ;

    int rc = LoadFromFile(path);

    char* load_digest = digest() ;

    LOG(LEVEL)
        << " items " << getItems()
        << " path " << path
        << " load_digest " << load_digest
        ;  

    m_dev_rng_states = copytodevice_rng_wrapper(m_launchseq, m_host_rng_states);
    devicesync();
     
    bool roundtrip = false ;
    if(roundtrip)
    {
       m_host_rng_states = copytohost_rng_wrapper(m_launchseq, m_dev_rng_states);
       devicesync();

       char* roundtrip_digest = digest();

       LOG(LEVEL)
           << " roundtrip_digest " << roundtrip_digest
           ; 

       assert(strcmp(load_digest, roundtrip_digest)==0);
       free(roundtrip_digest);
    }

    free(load_digest);

    LOG(LEVEL) << "]"  ; 
    return rc ;
}


int cuRANDWrapper::hasCache(const char* path)
{
    FILE* fp = fopen(path, "rb");
    if (fp) 
    {
        fclose(fp);
        return 1;
    }
    return 0;
}

/**
cuRANDWrapper::SaveToFile
----------------------------

Write getItems() curandState structs (m_host_rng_states) to file.

**/

int cuRANDWrapper::SaveToFile(const char* path)
{
    if(m_cache_dir && !hasCache(path))
    {
        LOG(LEVEL)
            << " create directory "
            << " path " << path      
            << " cache_dir " << m_cache_dir
            ;      

        mkdirp(m_cache_dir, 0777);
    }

    FILE *fp = fopen(path,"wb");
    if(fp == NULL) {
        LOG(fatal) << " error opening file " << path ; 
        return 1 ;
    }
    for(unsigned i = 0 ; i < getItems() ; ++i )
    {
        curandState& rng = m_host_rng_states[i] ;
        fwrite(&rng.d,                     sizeof(unsigned int),1,fp);
        fwrite(&rng.v,                     sizeof(unsigned int),5,fp);
        fwrite(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);
        fwrite(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);
    }
    fclose(fp);
    return 0 ;
}

/**
cuRANDWrapper::LoadFromFile
-----------------------------

Reads getItems() curandState structs (m_host_rng_states) from file.

**/

int cuRANDWrapper::LoadFromFile(const char* path)
{
    FILE *fp = fopen(path,"rb");
    if(fp == NULL) {
        LOG(fatal) << " error opening file " << path ; 
        return 1 ;
    }

    free(m_host_rng_states);
    m_host_rng_states = (curandState*)malloc(sizeof(curandState)*getItems());

    for(unsigned i = 0 ; i < getItems() ; ++i )
    {
        curandState& rng = m_host_rng_states[i] ;
        fread(&rng.d,                     sizeof(unsigned int),1,fp);
        fread(&rng.v,                     sizeof(unsigned int),5,fp);
        fread(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);
        fread(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);
        fread(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);
        fread(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);
    }
    fclose(fp);
    return 0 ;
}


/**
cuRANDWrapper::LoadIntoHostBuffer
----------------------------------

This is invoked by ORng::init from OPropagator::OPropagator when there is no --mask active

1. loads from cache file into m_host_rng_states
2. memcpy the states to the pointer provided in the argument

**/


int cuRANDWrapper::LoadIntoHostBuffer(curandState* host_rng_states, unsigned elements)
{
    LOG(LEVEL) << "[" ;  

    assert( elements == getItems()); 

    assert(hasCacheEnabled());

    const char* path = getCachePath() ;

    if(hasCache())
    {
        LOG(LEVEL) << " loading from cache " << path ; 

        int rc = LoadFromFile(path);

        assert(rc == 0);

        char* load_digest = digest() ;

        LOG(LEVEL) 
            << " items " << getItems() 
            << " path " <<  path
            << " load_digest " << load_digest
            ;      

        memcpy((void*)host_rng_states, (void*)m_host_rng_states, sizeof(curandState)*getItems());

    }
    else
    {
        LOG(error)
            << " MISSING RNG CACHE " << path  << std::endl 
            << " create with bash functions cudarap-;cudarap-prepare-installcache " << std::endl 
            << " should have been invoked by opticks-prepare-installcache "   
            ;  

        assert(0);
    }
    LOG(LEVEL) << "]" ;  
    return 0 ; 
}

/**
cuRANDWrapper::LoadIntoHostBufferMasked
-------------------------------------------

Invoked by ORng::init from OPropagator::OPropagator when a --mask is active

1. loads m_host_rng_states curandStates from cache
2. copies just the masked states into the argument pointer 

Hence this fabricates a GPU buffer with the curandStates needed just 
for the mask list of photon indices. 

**/

int cuRANDWrapper::LoadIntoHostBufferMasked(curandState* host_rng_states, const std::vector<unsigned>& mask)
{
    assert( hasCacheEnabled() && hasCache() );
    const char* path = getCachePath() ;
    int rc = LoadFromFile(path);
    assert(rc == 0);

    unsigned num_items = getItems() ;     
    unsigned num_mask = mask.size(); 

    LOG(LEVEL)
        << " num_items " << num_items 
        << " num_mask " << num_mask
        ; 

    char* sbytes = (char*)m_host_rng_states ;
    char* dbytes = (char*)host_rng_states ; 
    unsigned size = sizeof(curandState) ;

    unsigned s(0);
    for(unsigned i=0 ; i < num_mask ; i++)
    {
        unsigned item_id = mask[i] ; 
        assert( item_id < num_items ); 
        memcpy( (void*)(dbytes+size*s), (void*)(sbytes+size*item_id), size );
        s += 1 ;  
    }
    return 0 ; 
}



/**
cuRANDWrapper::resize
-----------------------

Hmm when the resize is decreasing in size could so this 
much more efficiently.

BUT: checking ORng this functionality is only ever used in tests


**/

void cuRANDWrapper::resize(unsigned elements)
{
    LOG(LEVEL) << " elements " << elements ; 
    if(getItems() == elements && !m_first_resize)
    {
        LOG(LEVEL) << " SKIP as size is unchanged " << elements ;
        return ;
    }

    setItems(elements);

    if(isOwner())
    {
       if(!m_first_resize) Free();
       Allocate();
    }

    InitFromCacheIfPossible();

    m_first_resize = false ; 
}


