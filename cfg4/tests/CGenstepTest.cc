
#include "OPTICKS_LOG.hh"
#include "CGenstep.hh"


void test_mask()
{
    unsigned index = 0 ; 
    unsigned photons = 100 ; 
    unsigned offset = 1000 ; 
    char gentype = 'C' ; 

    CGenstep gs(index, photons, offset, gentype); 

#ifdef WITH_CGENSTEP_MASK
    for(int i=0 ; i < int(photons) ; i++) if(i % 10 == 0 ) gs.set(i) ; 
#endif

    LOG(info) << gs.desc() ;
}

void test_assign()
{
    CGenstep gs ; 

    LOG(info) << gs.desc(); 

    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ; 
    for(unsigned i=0 ; i < strlen(s) ; i++)
    {
        gs = {i,i*10,i,s[i] } ; 
        LOG(info) << gs.desc(); 
    }

}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
 
    test_mask(); 
    test_assign(); 

    return 0 ; 
}

// om- ; TEST=CGenstepTest om-t
