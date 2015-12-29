#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchCommon.hh"
#include "LaunchSequence.hh"

#include "curand_kernel.h"
#include "md5digest.hh"
#include "stdio.h"
#include "assert.h"


cuRANDWrapper::cuRANDWrapper( LaunchSequence* launchseq, unsigned long long seed, unsigned long long offset )
           :
           m_launchseq(launchseq),
           m_seed(seed),
           m_offset(offset),
           m_dev_rng_states(0),
           m_host_rng_states(0),
           m_test(0),
           m_imod(100000),
           m_cache_dir(0),
           m_cache_enabled(true),
           m_owner(false),
           m_first_resize(true)
{
    setCacheDir("/tmp"); 
}

void cuRANDWrapper::setCacheDir(const char* dir)
{
    m_cache_dir = strdup(dir);
}
 
void cuRANDWrapper::setItems(unsigned int items)
{
    m_launchseq->setItems(items); 
}
unsigned int cuRANDWrapper::getItems()
{ 
    return m_launchseq->getItems() ; 
}

char* cuRANDWrapper::digest()
{
    if(!m_host_rng_states) return 0 ;

    MD5Digest dig ;
    dig.update( (char*)m_host_rng_states, sizeof(curandState)*getItems()) ; 
    return dig.finalize();
}

char* cuRANDWrapper::testdigest()
{
    MD5Digest dig ;
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

void cuRANDWrapper::test_rng(const char* tag)
{
    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag(tag);

    unsigned int items = seq->getItems();

    m_test  = (float*)malloc(items*sizeof(float));

    test_rng_wrapper( seq, m_dev_rng_states, m_test);

    char* test_digest = testdigest();
    printf("%s %s ", tag, test_digest);
    for(unsigned int i=0 ; i<items ; ++i)
    {
        if( i % m_imod == 0 )
        printf("%10.4f ", m_test[i] );
    } 
    printf("\n");


    free(test_digest);
    free(m_test);

    m_launchrec.push_back(seq); 
}


void cuRANDWrapper::Summary(const char* msg)
{
    unsigned int nrec = m_launchrec.size();
    for(unsigned int i=0 ; i < nrec ; i++)
    {
        LaunchSequence* seq = m_launchrec[i];  
        seq->Summary(msg); 
    }
}


char* cuRANDWrapper::getCachePath()
{
    char buf[256];
    snprintf(buf, 256, "%s/cuRANDWrapper_%u_%llu_%llu.bin", 
                 m_cache_dir,
                 getItems(),
                 m_seed,
                 m_offset); 
    return strdup(buf);
}


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

        // by inspection boxmuller_extra boxmuller_extra_double seem un-initialized
        // with flags unset, from docs appears to only be used for
        // curand_normal calls 
    }
}


int cuRANDWrapper::InitFromCacheIfPossible()
{
    printf("cuRANDWrapper::InitFromCacheIfPossible\n");
    if(!hasCacheEnabled())
    {
        printf("cuRANDWrapper::InitFromCacheIfPossible cache is disabled\n");
        Init();
    }
    else
    {
        if(hasCache())
        {
            printf("cuRANDWrapper::InitFromCacheIfPossible : loading from cache \n");
            Load();
        }
        else
        {
            printf("cuRANDWrapper::InitFromCacheIfPossible : no cache initing and saving \n");
            Init();
            Save();
        }
    }
    return 0 ;
}




int cuRANDWrapper::hasCache()
{
    char* path = getCachePath() ;
    int rc = hasCache(path);
    free(path);
    return rc ; 
}



int cuRANDWrapper::Allocate()
{
    printf("cuRANDWrapper::Allocate\n");
    m_owner = true ; 
    m_dev_rng_states = allocate_rng_wrapper(m_launchseq);
    devicesync();  
    return 0 ;
}

int cuRANDWrapper::Free()
{
    printf("cuRANDWrapper::Free\n");
    assert(isOwner());

    free_rng_wrapper(m_dev_rng_states);
    devicesync();  
    return 0 ;
}

int cuRANDWrapper::Init()
{
    printf("cuRANDWrapper::Init\n");

    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag("init");
    init_rng_wrapper( seq, m_dev_rng_states, m_seed, m_offset);
    m_launchrec.push_back(seq); 

    devicesync();  
    return 0 ;
}

int cuRANDWrapper::Save()
{
    printf("cuRANDWrapper::Save\n");
    char* path = getCachePath() ;

    m_host_rng_states = copytohost_rng_wrapper(m_launchseq, m_dev_rng_states);

    devicesync();

    char* save_digest = digest() ;

    printf("cuRANDWrapper::Save %u items to %s save_digest %s \n", getItems(), path, save_digest);

    int rc = Save(path);

    free(save_digest);
    free(path);
    return rc ; 
}


int cuRANDWrapper::LoadIntoHostBuffer(curandState* host_rng_states, unsigned int elements)
{
    printf("cuRANDWrapper::LoadIntoHostBuffer\n");

    assert( elements == getItems()); 

    if(!hasCacheEnabled())
    {
        printf("cuRANDWrapper::LoadIntoHostBuffer cache is disabled\n");
        assert(0);
    }
    else
    {
        char* path = getCachePath() ;
        if(hasCache())
        {
            printf("cuRANDWrapper::LoadIntoHostBuffer : loading from cache %s \n", path);

            int rc = Load(path);
            char* load_digest = digest() ;
            printf("cuRANDWrapper::LoadIntoHostBuffer %u items from %s load_digest %s \n", getItems(), path, load_digest);
            memcpy((void*)host_rng_states, (void*)m_host_rng_states, sizeof(curandState)*getItems());

        }
        else
        {
            printf("cuRANDWrapper::LoadIntoHostBuffer : no cache %s, initing and saving \n", path);
            assert(0);
        }
    }
 



    return 0 ; 
}



int cuRANDWrapper::Load()
{
    printf("cuRANDWrapper::Load\n");
    char* path = getCachePath() ;

    int rc = Load(path);

    char* load_digest = digest() ;
    printf("cuRANDWrapper::Load %u items from %s load_digest %s \n", getItems(), path, load_digest);

    m_dev_rng_states = copytodevice_rng_wrapper(m_launchseq, m_host_rng_states);
    devicesync();
     
    bool roundtrip = true ;
    if(roundtrip)
    {
       m_host_rng_states = copytohost_rng_wrapper(m_launchseq, m_dev_rng_states);
       devicesync();

       char* roundtrip_digest = digest();
       printf("cuRANDWrapper::Load roundtrip_digest %s \n", roundtrip_digest ); 
       assert(strcmp(load_digest, roundtrip_digest)==0);
       free(roundtrip_digest);
    }

    free(load_digest);
    free(path);   // getStatesPath returns need freeing

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

int cuRANDWrapper::Save(const char* path)
{
    if(m_cache_dir && !hasCache(path))
    {
        printf("cuRANDWrapper::Save mkdirp for path %s m_cache_dir %s \n", path, m_cache_dir );
        mkdirp(m_cache_dir, 0777);
    }

    FILE *fp = fopen(path,"wb");
    if(fp == NULL) {
        printf("cuRANDWrapper::Save error opening file %s \n", path);
        return 1 ;
    }
    for(unsigned int i = 0 ; i < getItems() ; ++i )
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


int cuRANDWrapper::Load(const char* path)
{
    FILE *fp = fopen(path,"rb");
    if(fp == NULL) {
        printf("cuRANDWrapper::Load ERROR opening file %s \n", path);
        return 1 ;
    }

    free(m_host_rng_states);
    m_host_rng_states = (curandState*)malloc(sizeof(curandState)*getItems());

    for(unsigned int i = 0 ; i < getItems() ; ++i )
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






cuRANDWrapper* cuRANDWrapper::instanciate(
         unsigned int elements, 
         const char* cachedir,
         unsigned long long seed,
         unsigned long long offset,
         unsigned int max_blocks,
         unsigned int threads_per_block
     )
{
    LaunchSequence* seq = new LaunchSequence( elements, threads_per_block, max_blocks ) ;
    cuRANDWrapper* crw = new cuRANDWrapper(seq, seed, offset);
    if(cachedir)
    {
        printf("cuRANDWrapper::instanciate with cache enabled : cachedir %s\n", cachedir);
        crw->setCacheDir(cachedir);
        crw->setCacheEnabled(true);
    }
    else
    {
        printf("cuRANDWrapper::instanciate with cache disabled\n");
        crw->setCacheEnabled(false);
    }
    return crw ; 
}



void cuRANDWrapper::resize(unsigned int elements)
{
    printf("cuRANDWrapper::resize\n");
    if(getItems() == elements && !m_first_resize)
    {
        printf("cuRANDWrapper::resize size is unchanged %u \n", elements);
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



int cuRANDWrapper::fillHostBuffer(curandState* host_rng_states, unsigned int elements)
{
    printf("cuRANDWrapper::fillHostBuffer\n");

    int rc = LoadIntoHostBuffer(host_rng_states, elements );

    return rc ;

}






