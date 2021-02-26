#include "NPY.hpp"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::vector<glm::mat4> mats ; 
    glm::mat4 m(1.f); 
    for(unsigned i=0 ; i < 10 ; i++) mats.push_back(m) ;

    NPY<float>* a = NPY<float>::make(mats.size(), 4, 4);
    a->read( mats.data() ); 

    const char* path = "$TMP/NPY7Test/mats.npy" ;
    LOG(info) << "saving " << path ; 
    a->save(path); 
    SSys::npdump(path, "np.float32"); 

    return 0 ; 
}
