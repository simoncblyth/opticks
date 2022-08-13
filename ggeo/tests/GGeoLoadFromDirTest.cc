#include <cstdlib>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "NPY.hpp"
#include "Opticks.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

const char* GGBASE = getenv("GGBASE") ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv, "--allownokey" );
    ok.configure(); 

    GGeo* ggeo = GGeo::LoadFromDir(&ok, GGBASE ); 
    unsigned nmm = ggeo->getNumMergedMesh();
    unsigned repeatIdx = SSys::getenvunsigned("RIDX", 1u)  ; 
    LOG(info) << " ggeo " << ggeo << " nmm " << nmm << " repeatIdx " << repeatIdx ; 

    assert( repeatIdx < nmm ); 

    const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx); 
    unsigned num_inst = mm->getNumITransforms() ; 
    NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();

    LOG(info) << " mm " << mm << " num_inst " << num_inst << " iid " << iid->getShapeString() ; 

    //std::vector<int> sensor_index ; 
    //mm->getInstancedIdentityBuffer_SensorIndex(sensor_index) ; 

    std::cout << mm->descInstancedIdentityBuffer_SensorIndex() << std::endl ; 

    return 0 ; 
}
