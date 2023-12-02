// ~/opticks/sysrap/tests/NP_delete_test.sh

#include "sprof.h"
#include "sstr.h"
#include "sstamp.h"

#include "NPFold.h"

struct NP_delete_test
{
    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static void t0(); 
    static void t1(); 
    static void t2(); 

    static void main(); 
};


/**
NP_delete_test::t0
---------------------

Looks like no cleanup ...

NP::descSize arr_bytes 64000000 arr_kb 64000
398833,73441,64095
279619,0,8

WHAT TO LEARN FROM THIS : DONT TRY CHECKING 
CLEANUP OF A SINGLE ACTION, THERES LOTS OF 
RUNTIME "PEDESTAL" TOO.  INSTEAD : DO THINGS 
IN A LOOP AND LOOK FOR VARIATIONS, SEE BELOW. 

**/

void NP_delete_test::t0()
{
    sprof p0, p1, p2 ; 

    sprof::Stamp(p0);  

    NP* a = NP::Make<float>( 1*M, 4, 4 ) ; 
    std::cout << a->descSize() << std::endl ; 

    sprof::Stamp(p1);  

    //a->clear() ; 

    a->data.clear() ; 
    a->data.shrink_to_fit();
    //delete a ; 
    //a = nullptr ; 

    sprof::Stamp(p2);  

    std::cout << sprof::Desc(p0,p1) << std::endl ;  
    std::cout << sprof::Desc(p1,p2) << std::endl ;  
}


/**
NP_delete_test::t1
--------------------

Without the delete the RSS keeps sloping upwards, with it it stays flat. 

YES: but using the same size every time is disceptive, that makes it 
easy for the system to allocate after the delete 

**/

void NP_delete_test::t1()
{
    const char* NumPhotonSpec = "M1,1,1,1,1" ; 
    std::vector<int>* nums = sstr::ParseIntSpecList<int>( NumPhotonSpec, ',' ) ; 
    int num_event = nums ? nums->size() : 0 ; 

    NP* run = NP::Make<int>(num_event); 
    int* rr = run->values<int>(); 

    run->set_meta<std::string>("TEST", "t1") ; 

    bool CLEAR = getenv("CLEAR") != nullptr ; 
    bool DELETE = getenv("DELETE") != nullptr ; 
    run->set_meta<std::string>("CLEAR", CLEAR ? "YES" : "NO" ) ; 
    run->set_meta<std::string>("DELETE", DELETE ? "YES" : "NO" ) ; 


    for(int idx=0 ; idx < 10 ; idx++)
    {
        std::string head = U::FormName_("head_", idx, nullptr, 3 ) ; 
        run->set_meta<std::string>(head.c_str(), sprof::Now() ); 

        int num = (*nums)[idx] ; 
        rr[idx] = num ;  

        NP* a = NP::Make<float>( num, 4, 4 ) ; 

        std::string body = U::FormName_("body_", idx, nullptr, 3 ) ; 
        run->set_meta<std::string>(body.c_str(), sprof::Now() ); 
 
        if(CLEAR) a->clear() ;     
        if(DELETE) delete a ; 

        std::string tail = U::FormName_("tail_", idx, nullptr, 3 ) ; 
        run->set_meta<std::string>(tail.c_str(), sprof::Now() ); 

        sstamp::sleep_us(100000); 
    }

    NPFold* fold = new NPFold ;  
    fold->add( "run", run ); 
    fold->add( "runprof", run->makeMetaKVProfileArray() ); 
    fold->save("$FOLD"); 
}




/**
NP_delete_test::t2
--------------------

Varying the sizes gives a much less flat RSS, but still 
not deleting allows it to grow it seems that when deleting 
are left with the largest size allocation. 

::

    In [5]: f.run*16*4/1e9   ## estmating GB for  "M1,2,3,1,2,3"
    Out[5]: array([0.064, 0.128, 0.192, 0.064, 0.128, 0.192])


**/

void NP_delete_test::t2()
{
    //const char* NumPhotonSpec = "H1,2,4,8,16" ; 
    const char* NumPhotonSpec = "M1,2,3,4,3,2,1" ; 
    std::vector<int>* nums = sstr::ParseIntSpecList<int>( NumPhotonSpec, ',' ) ; 
    
    int num_event = nums ? nums->size() : 0 ; 

    NP* run = NP::Make<int>(num_event); 
    int* rr = run->values<int>(); 

    run->set_meta<std::string>("TEST", "t2" ) ; 

    bool CLEAR = getenv("CLEAR") != nullptr ; 
    bool DELETE = getenv("DELETE") != nullptr ; 
    run->set_meta<std::string>("CLEAR", CLEAR ? "YES" : "NO" ) ; 
    run->set_meta<std::string>("DELETE", DELETE ? "YES" : "NO" ) ; 


    for(int idx=0 ; idx < num_event  ; idx++)
    {
        int num = (*nums)[idx] ; 
        rr[idx] = num ;  
        std::cout << std::setw(4) << idx << " : " << num << std::endl ; 

        std::string head = U::FormName_("head_", idx, nullptr, 3 ) ; 
        sstamp::sleep_us(100000); 
        run->set_meta<std::string>(head.c_str(), sprof::Now() ); 

        NP* a = NP::Make<float>( num, 4, 4 ) ; 


        std::string body = U::FormName_("body_", idx, nullptr, 3 ) ; 
        sstamp::sleep_us(100000); 
        run->set_meta<std::string>(body.c_str(), sprof::Now() ); 

        if(CLEAR) a->clear() ;     
        if(DELETE) delete a ; 


        std::string tail = U::FormName_("tail_", idx, nullptr, 3 ) ; 
        sstamp::sleep_us(100000); 
        run->set_meta<std::string>(tail.c_str(), sprof::Now() ); 
    }

    NPFold* fold = new NPFold ;  
    fold->add( "run", run ); 
    fold->add( "runprof", run->makeMetaKVProfileArray() ); 
    fold->save("$FOLD"); 

}





void NP_delete_test::main()
{
    char* TEST = getenv("TEST") ; 
    int test = TEST && strlen(TEST) > 0 && TEST[0] == 't'  ? strtol(TEST+1, nullptr, 10) : -1 ; 
    switch(test)
    {
        case 0: t0() ; break ; 
        case 1: t1() ; break ; 
        case 2: t2() ; break ; 
        case 3: t3() ; break ; 
    }
}

int main()
{
    NP_delete_test::main(); 
    return 0 ; 
}

// ~/opticks/sysrap/tests/NP_delete_test.sh

