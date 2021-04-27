#include "NPY.hpp"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"

NPY<float>* test_make()
{
    std::vector<glm::mat4> mats ; 
    glm::mat4 m(1.f); 
    for(unsigned i=0 ; i < 10 ; i++) mats.push_back(m) ;
    NPY<float>* a = NPY<float>::make(mats.size(), 4, 4);
    a->read( mats.data() ); 

    const char* path = "$TMP/NPY7Test/mats.npy" ;
    LOG(info) << "saving " << path ; 
    a->save(path); 
    SSys::npdump(path, "np.float32"); 

    return a ; 
}


void test_save_nulldir(const NPY<float>* a )
{
    a->save(nullptr, "test_save_nulldir.npy" );   // fails 
    //a->save(".", "test_save_nulldir.npy" );     // works
    //a->save("test_save_nulldir.npy" );          // fails
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NPY<float>* a = test_make(); 
    test_save_nulldir(a); 

    return 0 ; 
}
