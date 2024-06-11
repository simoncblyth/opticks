/**

  ~/o/sysrap/tests/slist_test.sh 

**/


#include "ssys.h"
#include "slist.h"

struct slist_test
{
    static int FindIndices(); 
    static int FindIndex(); 
    static int FindIndex_0(); 
    static int FindIndex_1(); 
    static int Contains(); 
    static int Main(); 
};

inline int slist_test::FindIndices()
{
    std::cerr << "slist_test::FindIndices\n" ; 
    std::vector<int> idx_red ; 
    std::vector<std::string> name = {{ "red", "green" , "blue", "red", "redgreen", "redblue" }} ; 
    slist::FindIndices(idx_red, name, "red" ); 
    assert( idx_red.size() == 2 && idx_red[0] == 0 && idx_red[1] == 3 ); 
    return 0 ; 
}

inline int slist_test::FindIndex()
{
    std::cerr << "slist_test::FindIndex\n" ; 
    int rc = 0 ; 
    rc += FindIndex_0(); 
    rc += FindIndex_1(); 
    return rc ; 
}

inline int slist_test::FindIndex_0()
{
    std::cerr << "slist_test::FindIndex_0\n" ; 
    std::vector<std::string> name = {{ "red", "green" , "blue", "red", "redgreen", "redblue" }} ; 
    int idx = slist::FindIndex(name, "red" ); 
    assert( idx == -1 ); 
    return 0 ; 
}

inline int slist_test::FindIndex_1()
{
    std::cerr << "slist_test::FindIndex_1\n" ; 
    std::vector<std::string> name = {{ "red", "green" , "blue", "red", "redgreen", "redblue" }} ; 
    int idx = slist::FindIndex(name, "green" ); 
    assert( idx == 1 ); 
    return 0 ; 
}

inline int slist_test::Contains()
{
    std::cerr << "slist_test::Contains\n" ; 
    std::vector<int> idx = {100, 200, 300} ; 
    bool q_101 = slist::Contains(idx, 101 ); 
    assert( q_101 == false );

    bool q_100 = slist::Contains(idx, 100 ); 
    assert( q_100 == true );

    return 0 ; 
}



inline int slist_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL" ); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    int rc = 0 ; 
    if(ALL || strcmp(TEST, "FindIndex")   == 0 ) rc += FindIndex(); 
    if(ALL || strcmp(TEST, "FindIndices") == 0 ) rc += FindIndices(); 
    if(ALL || strcmp(TEST, "Contains")    == 0 ) rc += Contains(); 
    return rc ; 
}

int main()
{
    return slist_test::Main() ; 
}
