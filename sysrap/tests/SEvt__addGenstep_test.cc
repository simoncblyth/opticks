// ~/opticks/sysrap/tests/SEvt__addGenstep_test.sh 


#include "scuda.h"
#include "squad.h"
#include "NPFold.h"


struct SEvt__addGenstep_test
{
    static void addGenstep(const NP* a)
    {
        int num_gs = a ? a->shape[0] : -1 ; 
        assert( num_gs > 0 ); 
        quad6* qq = (quad6*)a->bytes(); 
        for(int i=0 ; i < num_gs ; i++) addGenstep(qq[i]) ; 
    }

    static void addGenstep(const quad6& q_)
    {
        unsigned gentype = q_.gentype(); 
        unsigned matline_ = q_.matline(); 
        std::cout << "addGenstep_Q6 " << matline_ << std::endl ; 
        quad6& q = const_cast<quad6&>(q_);
        q.set_matline(100);  
        // THIS IS CHANGING THE ORIGIN ARRAY 
    }
};


int main(int argc, char** argv)
{
    const int N = 2 ; 
    std::vector<quad6> gs(N) ; 
    for(int i=0 ; i < N ; i++) gs[i].zero() ; 
    const NP* a = NPX::ArrayFromVec<int,quad6>( gs, 6, 4) ; 
    NP* a0 = a->copy(); 

    SEvt__addGenstep_test::addGenstep( a ); 

    NP* a1 = a->copy(); 


    NPFold* f = new NPFold ; 
    f->add("a0", a0 ); 
    f->add("a1", a1 ); 
    f->save("$FOLD"); 
    
    return 0 ; 
}


