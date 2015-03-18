#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchCommon.hh"
#include "LaunchSequence.hh"

#include "curand_kernel.h"
#include "md5digest.hh"
#include "stdio.h"
#include "assert.h"


unsigned int cuRANDWrapper::getItems()
{ 
    return m_launchseq->getItems() ; 
}

void cuRANDWrapper::create_rng()
{
    m_dev_rng_states = create_rng_wrapper(m_launchseq);
}

void cuRANDWrapper::copytohost_rng()
{
    m_host_rng_states = copytohost_rng_wrapper(m_launchseq, m_dev_rng_states);
}

void cuRANDWrapper::copytodevice_rng()
{
    //TODO: avoid leaks 
    m_dev_rng_states = copytodevice_rng_wrapper(m_launchseq, m_host_rng_states);
}


char* cuRANDWrapper::digest()
{
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



void cuRANDWrapper::init_rng(const char* tag)
{
    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag(tag);

    init_rng_wrapper(
        seq,
        m_dev_rng_states, 
        m_seed, 
        m_offset
    );

    m_launchrec.push_back(seq); 
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


void cuRANDWrapper::test_rng(const char* tag)
{
    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag(tag);

    unsigned int items = seq->getItems();

    m_test  = (float*)malloc(items*sizeof(float));

    test_rng_wrapper(
        seq, 
        m_dev_rng_states, 
        m_test
    );

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

void cuRANDWrapper::setCacheDir(const char* dir)
{
    m_cache_dir = strdup(dir);
}
cuRANDWrapper::~cuRANDWrapper()
{
    free(m_cache_dir);
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


void cuRANDWrapper::Setup(bool create)
{
    if(hasCache())
    {
        Load();
        //Dump("loaded_from_cache",100000);
    }
    else
    {
        Init(create);
        Save();
        //Dump("init_and_cached",100000);
    }
}


int cuRANDWrapper::hasCache()
{
    char* path = getCachePath() ;
    int rc = hasCache(path);
    free(path);
    return rc ; 
}

int cuRANDWrapper::Init(bool create)
{
    if(create)
    {
        create_rng();
    }
    init_rng("init");
    return 0 ;
}

int cuRANDWrapper::Save()
{
    char* path = getCachePath() ;
    printf("cuRANDWrapper::Save %u items to %s \n", getItems(), path);

    copytohost_rng();
    int rc = Save(path);

    free(path);
    return rc ; 
}

int cuRANDWrapper::Load(bool roundtrip)
{
    char* path = getCachePath() ;

    int rc = Load(path);

    char* load_digest = digest() ;
    printf("cuRANDWrapper::Load %u items from %s load_digest %s \n", getItems(), path, load_digest);

    copytodevice_rng();
     
    if(roundtrip)
    {
       copytohost_rng();
       char* roundtrip_digest = digest();
       //printf("cuRANDWrapper::Load roundtrip_digest %s \n", roundtrip_digest ); 
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
        printf("cuRANDWrapper::Save error opening file %s", path);
        return 1 ;
    }
    curandState* rng_states = (curandState*)m_host_rng_states ; 
    for(unsigned int i = 0 ; i < getItems() ; ++i )
    {
        curandState& rng = rng_states[i] ;
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
        printf("cuRANDWrapper::Load ERROR opening file %s", path);
        return 1 ;
    }

    free(m_host_rng_states);
    m_host_rng_states = malloc(sizeof(curandState)*getItems());
    curandState* rng_states = (curandState*)m_host_rng_states ; 

    for(unsigned int i = 0 ; i < getItems() ; ++i )
    {
        curandState& rng = rng_states[i] ;
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




