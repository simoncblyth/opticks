#include "LaunchCommon.hh"
#include "LaunchSequence.hh"
#include "cuRANDWrapper.hh"

//#define WORK 1024*768
#define WORK 1024*1


int exists(const char *fname)
{
    FILE* fp = fopen(fname, "rb");
    if (fp) 
    {
        fclose(fp);
        return 1;
    }
    return 0;
}


int main(int argc, char** argv)
{
    unsigned int work = getenvvar("WORK", WORK) ;
    unsigned int threads_per_block = getenvvar("THREADS_PER_BLOCK", 256) ; 
    unsigned int max_blocks = getenvvar("MAX_BLOCKS", 128) ;
    bool reverse = false ; 


    LaunchSequence* seq = new LaunchSequence( work, threads_per_block, max_blocks, reverse) ;
    seq->Summary("seq");

    cuRANDWrapper* crw = new cuRANDWrapper(seq);

    char path[128];
    snprintf( path, 128, "/tmp/rng_states_%u.bin", work) ; 

    if(exists(path))
    {
        crw->Load(path);
        crw->Dump("loaded", 100000);
        crw->copytodevice_rng();
    }
    else
    {
        crw->create_rng();
        crw->init_rng("init");

        crw->copytohost_rng();
        crw->Dump("created",100000);
        crw->Save(path);
    }


    LaunchSequence* tseq = seq->copy(max_blocks*32); // can increase max_blocks as test_rng much faster than init_rng 
    crw->setLaunchSequence(tseq);

    crw->test_rng("test_0");
    crw->test_rng("test_1");
    crw->test_rng("test_2");
    crw->test_rng("test_3");
    crw->test_rng("test_4");

    crw->Summary("crw");

}

