#include <cstdlib>
#include "OPTICKS_LOG.hh"

#include "stree.h"
#include "SSys.hh"
#include "SVec.hh"
#include "NP.hh"

#include "NPY.hpp"
#include "Opticks.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

const char* BASE = getenv("BASE") ; 


std::string desc_iid_SensorIndex( const GGeo* ggeo, int ridx, const stree* st )
{
    const GMergedMesh* mm = ggeo->getMergedMesh(ridx); 
    unsigned num_inst = mm->getNumITransforms() ; 
    NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();

    std::vector<int> sensor_index ; 
    mm->getInstancedIdentityBuffer_SensorIndex(sensor_index) ; 

    bool one_based_index = true ; 
    std::vector<int> sensor_id ; 
    st->lookup_sensor_identifier(sensor_id, sensor_index, one_based_index); 


    int idx_mn, idx_mx ; 
    SVec<int>::MinMax(sensor_index, idx_mn, idx_mx ); 

    int id_mn, id_mx ; 
    SVec<int>::MinMax(sensor_id, id_mn, id_mx ); 

    std::stringstream ss ; 
    ss 
        << " ridx " << std::setw(3) << ridx 
        << " mm " << std::setw(10) << mm 
        << " num_inst " << std::setw(7) << num_inst 
        << " iid " << std::setw(15) << iid->getShapeString() 
        << " sensor_index " << std::setw(7) << sensor_index.size()
        << " idx_mn " << std::setw(7) << idx_mn
        << " idx_mx " << std::setw(7) << idx_mx
        << " id_mn " << std::setw(7) << id_mn
        << " id_mx " << std::setw(7) << id_mx
        ; 

    //ss << mm->descInstancedIdentityBuffer_SensorIndex() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv, "--allownokey" );
    ok.configure(); 

    GGeo* ggeo = GGeo::LoadFromDir(&ok, BASE ); 
    unsigned nmm = ggeo->getNumMergedMesh();

    //stree* st = stree::Load(BASE);   // after regrab can use this 
    stree* st = stree::Load(BASE, "stree_reorderSensors"); 

    std::cout << st->desc_sensor_id() << std::endl ;

    
    /*
    NP* sensor_id = NP::Load("/tmp/blyth/opticks/ntds3/G4CXOpticks/stree_reorderSensors/sensor_id.npy") ; 
    const int* sid = sensor_id->cvalues<int>(); 
    unsigned num_sid = sensor_id->shape[0] ; 
    */

    int ridx = SSys::getenvint("RIDX", -1); 
    LOG(info) << " ggeo " << ggeo << " nmm " << nmm << " ridx " << ridx ; 
    assert( ridx < int(nmm) ); 


    if( ridx > -1 )
    {
        std::cout << desc_iid_SensorIndex(ggeo, ridx, st) << std::endl ; 
    }
    else
    {
        for(unsigned ridx=0 ; ridx < nmm ; ridx++)
        {
            std::cout << desc_iid_SensorIndex(ggeo, ridx, st) << std::endl ; 
        }
    }

    return 0 ; 
}
