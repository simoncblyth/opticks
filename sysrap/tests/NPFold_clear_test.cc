/**
NPFold_clear_test.sh
======================

::

    ~/opticks/sysrap/tests/NPFold_clear_test.sh

Without the NPFold::clear call get assert::

    NPFold::add_ FATAL : have_key_already [photon.npy]

**/


#include "sprof.h"
#include "ssys.h"
#include "sstamp.h"

#include "NPFold.h"
#include "NP.hh"


struct SEV
{
    static constexpr const char* PHOTON = "photon" ; 
    static constexpr const char* SAVE_COMP = "photon" ; 
    static std::vector<int>* NUM ; 
    static int GetNumEvt(); 
    static int GetNum(int idx); 

    int idx ; 
    NPFold* fold ; 

    SEV(); 

    NP* gatherPhoton() const ; 
    void gather_components() ; 
    void save(const char* dir); 
    void clear(); 
};

std::vector<int>* SEV::NUM = ssys::getenv_ParseIntSpecList("NUM", "M1,2,3,4,3,2,1" ); 

int SEV::GetNumEvt()
{
    return NUM ? NUM->size() : 0 ; 
}
int SEV::GetNum(int idx) // 
{
    return NUM && idx < int(NUM->size()) ? (*NUM)[idx] : -1 ; 
}

inline SEV::SEV()
    :
    idx(-1),
    fold(new NPFold)
{
}

inline NP* SEV::gatherPhoton() const 
{
    assert( idx > -1 ); 
    int num = GetNum(idx); 
    NP* a = NP::Make<float>( num, 4, 4 ) ; 
    return a ;  
}

/**
SEV::gather_components
------------------------

NPFold::add asserts that the key is not already present 

**/

inline void SEV::gather_components()
{
    NP* a = gatherPhoton(); 
    fold->add(PHOTON, a ) ;   // NB the fold must have been cleared for this to work beyond first cycle 
}

inline void SEV::save(const char* dir)
{
    gather_components(); 
    bool shallow = true ;   // no ownership, just copy pointers 
    NPFold* save_fold = fold->copy(SAVE_COMP, shallow) ;
    save_fold->save(dir);
}
inline void SEV::clear()
{
    fold->clear();  // deletes the arrays 
}


struct NPFold_clear_test
{
    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static void t0(); 
    static void main(); 
};


void NPFold_clear_test::t0()
{

    bool CLEAR = getenv("CLEAR") != nullptr ; 
    bool DELETE = getenv("DELETE") != nullptr ; 

    int num_event = SEV::GetNumEvt(); 

    NP* run = NP::Make<int>(num_event); 
    run->set_meta<std::string>("TEST", "t0" ) ; 
    run->set_meta<std::string>("CLEAR", CLEAR ? "YES" : "NO" ) ; 
    run->set_meta<std::string>("DELETE", DELETE ? "YES" : "NO" ) ; 
    int* rr = run->values<int>(); 

    SEV* ev = new SEV ; 


    for(int idx=0 ; idx < num_event  ; idx++)
    {
        ev->idx = idx ; 
        int num = SEV::GetNum(idx); 
        rr[idx] = num ;  
        std::cout << std::setw(4) << idx << " : " << num << std::endl ; 

        std::string head = U::FormName_("head_", idx, nullptr, 3 ) ; 
        sstamp::sleep_us(100000); 
        run->set_meta<std::string>(head.c_str(), sprof::Now() ); 

        std::string dir = sstr::FormatIndexDefault_( idx, "$FOLD/");  
        ev->save(dir.c_str()) ;  

        std::string body = U::FormName_("body_", idx, nullptr, 3 ) ; 
        sstamp::sleep_us(100000); 
        run->set_meta<std::string>(body.c_str(), sprof::Now() ); 

        ev->clear();      

        std::string tail = U::FormName_("tail_", idx, nullptr, 3 ) ; 
        sstamp::sleep_us(100000); 
        run->set_meta<std::string>(tail.c_str(), sprof::Now() ); 
    }


    NPFold* out = new NPFold ;  
    out->add( "run", run ); 
    out->add( "runprof", run->makeMetaKVProfileArray() ); 
    out->save("$FOLD"); 
}

void NPFold_clear_test::main()
{
    char* TEST = getenv("TEST") ; 
    int test = TEST && strlen(TEST) > 0 && TEST[0] == 't'  ? strtol(TEST+1, nullptr, 10) : -1 ; 
    switch(test)
    {
        case 0: t0() ; break ; 
    }
}

int main()
{
    NPFold_clear_test::main(); 
    return 0 ; 
}

// ~/opticks/sysrap/tests/NPFold_clear_test.sh

