#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchSequence.hh"

#include "curand_kernel.h"
#include "md5digest.hh"

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
    unsigned int items = m_launchseq->getItems();
    size_t nbytes = sizeof(curandState)*items ; 
    MD5Digest dig ;
    dig.update( (char*)m_host_rng_states, nbytes);
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

    for(unsigned int i=0 ; i<items ; ++i)
    {
        if( i % m_imod == 0 )
        printf("%7u %10.4f \n", i, m_test[i] );
    } 

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

void cuRANDWrapper::Dump(const char* msg, unsigned int imod)
{
    char* dig = digest() ;
    printf("%s digest %s \n", msg, dig);
    free(dig);

    curandState* rng_states = (curandState*)m_host_rng_states ;
    unsigned int items = m_launchseq->getItems();
    for(unsigned int i = 0 ; i < items ; ++i )
    {
        if(i % imod != 0) continue ;   
        curandState& rng = rng_states[i] ;
        printf("d %10u v %10u %10u %10u %10u %10u boxmuller_extra %10.4f _extra_double %10.4f \n", 
            rng.d, 
            rng.v[0], 
            rng.v[1], 
            rng.v[2], 
            rng.v[3], 
            rng.v[4],
            rng.boxmuller_flag        ? rng.boxmuller_extra        : -1.f , 
            rng.boxmuller_flag_double ? rng.boxmuller_extra_double : -1.f );

        // by inspection boxmuller_extra boxmuller_extra_double seem un-initialized
        // with flags unset 
    }
}

void cuRANDWrapper::Save(const char* path)
{
    FILE *fp = fopen(path,"wb");
    if(fp == NULL) {
        printf("cuRANDWrapper::Save error opening file %s", path);
        return ;
    }
    unsigned int items = m_launchseq->getItems();
    printf("cuRANDWrapper::Save %u items to %s \n", items, path);

    curandState* rng_states = (curandState*)m_host_rng_states ; 
    for(unsigned int i = 0 ; i < items ; ++i )
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
}


void cuRANDWrapper::Load(const char* path)
{
    FILE *fp = fopen(path,"rb");
    if(fp == NULL) {
        printf("cuRANDWrapper::Load error opening file %s", path);
        return ;
    }
    unsigned int items = m_launchseq->getItems();

    free(m_host_rng_states);
    m_host_rng_states = malloc(sizeof(curandState)*items);

    printf("cuRANDWrapper::Load %u items from %s \n", items, path);

    curandState* rng_states = (curandState*)m_host_rng_states ; 
    for(unsigned int i = 0 ; i < items ; ++i )
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
}




