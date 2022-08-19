// ./stree_sensor_test.sh 

#include <cstdlib>
#include <iostream>
#include "stree.h"

const char* BASE = getenv("BASE");  

void test_lookup_sensor_identifier( const stree& st )
{
    std::vector<int> s_ident ; 
    const std::vector<int> s_index = {{ 
          -4,-3,-2,-1,0,1,2,3,4, 
          1000, 
          10000, 
          17610, 17611, 17612, 17613, 17614, 17615, 17616, 
          20000, 
          30000, 
          40000, 
          43210, 43211, 43212, 43213, 43214, 43215, 43216, 
          45610, 45611, 45612, 45613, 45614, 45615, 45616, 
          50000, 50001, 50002 }} ;  

    bool one_based_index = true ; 
    bool verbose = true ; 
    unsigned edge = 50 ; 

    st.lookup_sensor_identifier( s_ident, s_index, one_based_index,  verbose, edge );  
 
    std::cout 
        << " test_lookup_sensor_identifier " 
        << " st.sensor_id.size " << st.sensor_id.size()
        << " s_index.size " << s_index.size()    
        << " s_ident.size " << s_ident.size()    
        << std::endl
        ;

    assert( s_index.size()  == s_ident.size() ); 

    for(unsigned i=0 ; i < s_index.size() ; i++ )
    {
         std::cout 
             << " i " << std::setw(6) << i 
             << " s_index[i] " << std::setw(6) << s_index[i]
             << " s_ident[i] " << std::setw(6) << s_ident[i]
             << std::endl 
             ;
    } 
}

void test_sensor_id( const stree& st )
{
    std::cout << " st.sensor_id.size " << st.sensor_id.size() << std::endl ; 
    std::cout << " st.sensor_id[st.sensor_id.size()-1] " << st.sensor_id[st.sensor_id.size()-1] << std::endl ; 

    // using std::vector "[]" does not range check and does not give any error, returning "garbage" values 
    std::cout << " st.sensor_id[st.sensor_id.size()] " << st.sensor_id[st.sensor_id.size()] << std::endl ; 
    std::cout << " st.sensor_id[st.sensor_id.size()+1] " << st.sensor_id[st.sensor_id.size()+1] << std::endl ; 
    std::cout << " st.sensor_id[st.sensor_id.size()+1000] " << st.sensor_id[st.sensor_id.size()+1000] << std::endl ; 
    std::cout << " st.sensor_id[st.sensor_id.size()+1001] " << st.sensor_id[st.sensor_id.size()+1001] << std::endl ; 

    // using std::vector ".at" does range check and throws : libc++abi.dylib: terminating with uncaught exception of type std::out_of_range: vector
    //std::cout << " st.sensor_id.at(st.sensor_id.size()) " << st.sensor_id.at(st.sensor_id.size()) << std::endl ; 
}


void test_labelFactorSubtrees( stree& st )
{
    st.level = 2 ; 
    st.labelFactorSubtrees();   // not normally called other than from factorize : but just testing here  
}



int main(int argc, char** argv)
{
    stree st ; 
    st.load(BASE); 

    std::cout << "st.desc_sub(false)" << std::endl << st.desc_sub(false) << std::endl ;
    std::cout << "st.desc_sensor_id()" << std::endl << st.desc_sensor_id() << std::endl ; 

    /*
    test_lookup_sensor_identifier( st );  
    test_sensor_id( st ); 
    */

    test_labelFactorSubtrees(st); 





    return 0 ; 
}
