/**
SCurandState_test.cc
=====================

~/o/sysrap/tests/SCurandState_test.sh 

NEXT: 
    implement loading of any number of curandState within the range 
    by deciding which chunks to load and typically doing 
    a partial load of the last chunk 
    (current range is 0->200M) 

**/

#include "ssys.h"
#include "SCurandState.h"


struct SCurandState_test
{
    static SCurandChunk Chunk(); 

    static int ctor();
    static int NumFromFilesize();
    static int ParseDir();
    static int ChunkLoadSave();
    static int load();
    static int Main();
};

inline SCurandChunk SCurandState_test::Chunk()
{
    SCurandChunk chunk = {} ;
    chunk.ref.chunk_idx = 0 ; 
    chunk.ref.chunk_offset = 0 ; 
    chunk.ref.num = 1000000 ; 
    chunk.ref.seed = 0 ; 
    chunk.ref.offset = 0 ; 

    return chunk ; 
}

inline int SCurandState_test::ctor()
{
    _SCurandState cs ; 
    std::cout << cs.desc(); 
    return 0 ; 
}

inline int SCurandState_test::NumFromFilesize()
{
    SCurandChunk chunk = Chunk(); 
    std::string n = chunk.name() ; 
    long st = SCurandChunk::NumFromFilesize(n.c_str()); 
    int rc = st == 1000000 ? 0 : 1 ; 
    return rc ; 
}

inline int SCurandState_test::ParseDir()
{
    std::vector<SCurandChunk> chunks ; 
    SCurandChunk::ParseDir(chunks ); 

    return 0 ; 
}

inline int SCurandState_test::ChunkLoadSave()
{
    SCurandChunk chunk0 = Chunk(); 
    std::string n0 = chunk0.name(); 
 
    typedef unsigned long long ULL ; 
    ULL q_num = 0 ; 
    const char* dir = nullptr ;  

    SCurandChunk chunk = {} ;
    int rc0 = SCurandChunk::OldLoad(chunk, n0.c_str(), q_num, dir) ;
    int rc1 = chunk.save("$FOLD"); 

    std::cout 
        << "SCurandState_test::ChunkLoadSave" 
        << " rc0 " << rc0 
        << " rc1 " << rc1 
        << "\n" 
        ; 

    return rc0 + rc1 ; 
}

inline int SCurandState_test::load()
{
    SCurandChunk c = Chunk(); 
    scurandref lref = c.load(); 

    std::cout 
        << "SCurandState_test::load\n"
        << " c.ref.desc " << c.ref.desc() << "\n"
        << " lref.desc  " << lref.desc() << "\n"
        << "\n" 
        ; 

    return lref.states ? 0 : 1 ; 
}


inline int SCurandState_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    int rc = 0 ;  
    if(ALL||strcmp(TEST,"ctor") == 0)            rc += ctor(); 
    if(ALL||strcmp(TEST,"NumFromFilesize") == 0) rc += NumFromFilesize(); 
    if(ALL||strcmp(TEST,"ParseDir") == 0)        rc += ParseDir(); 
    if(ALL||strcmp(TEST,"ChunkLoadSave") == 0)   rc += ChunkLoadSave(); 
    if(ALL||strcmp(TEST,"load") == 0)            rc += load(); 
    return rc ; 
}


int main()
{
    return SCurandState_test::Main() ;
}
