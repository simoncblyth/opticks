#include "NP.hh"
#include "OpticksRandom.hh"

struct OpticksRandomTest
{
    const NP* seq ; 
    OpticksRandom r ; 
    OpticksRandomTest(const char* path); 
    void basics(); 
};

OpticksRandomTest::OpticksRandomTest(const char* path)
    :
    seq(NP::Load(path)),
    r(seq)
{
}

void OpticksRandomTest::basics()
{
    r.dump(); 
    r.setSequenceIndex(0); 
    r.dump(); 
    r.setSequenceIndex(-1); 
    r.dump(); 
}


int main()
{
    OpticksRandomTest t("/tmp/blyth/opticks/TRngBufTest_0.npy"); 
    t.basics(); 
    return 0 ; 
}

