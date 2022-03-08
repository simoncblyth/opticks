
#include <vector>
#include <algorithm>
#include "SEnabled.hh"
#include "OPTICKS_LOG.hh"


template<unsigned N>
void test_isEnabled(const char* spec, const std::vector<unsigned>& e)
{
    SEnabled<N>* idx = new SEnabled<N>(spec);  
    for(unsigned i=0 ; i < N ; i++)
    {
        bool expect = std::find(e.begin(), e.end(), i) != e.end() ; 
        bool enabled = idx->isEnabled(i) ; 
        assert( expect == enabled );  
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_isEnabled<16>("10,5,11", {10,5,11} ); 
    test_isEnabled<16>("10,5,-1", {10,5,15} ); 

    test_isEnabled<1024>("1001,101,11,-1", {1001,101,11,1023} ); 

    return 0 ; 
}
