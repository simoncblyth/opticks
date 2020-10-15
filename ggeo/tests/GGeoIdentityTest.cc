#include <iostream>
#include <iomanip>

#include "SStr.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"
#include "GVector.hh"
#include "GMergedMesh.hh"


/**

   getIdentity 

   getInstancedIdentity



**/


int test_GGeo_identity(const GGeo* gg, unsigned mmidx)
{
    const GMergedMesh* mm = gg->getMergedMesh(mmidx);
    unsigned numVolumes = mm->getNumVolumes();
    unsigned numVolumesSelected = mm->getNumVolumesSelected();  // probably not persisted

    NPY<int>* idchk = NPY<int>::make(numVolumes,3,4) ; 
    idchk->zero(); 


    unsigned edgeitems = 20 ; 
    unsigned modulo = 500 ; 
    LOG(info) 
        << " mmidx " << mmidx 
        << " numVolumes " << numVolumes 
        << " numVolumesSelected " << numVolumesSelected 
        << " edgeitems " << edgeitems 
        << " modulo " << modulo 
        ;
 
    for(unsigned i=0 ; i < numVolumes ; i++)
    {
        guint4 nodeinfo = mm->getNodeInfo(i);
        guint4 id = mm->getIdentity(i);

        guint4 iid = mm->getInstancedIdentity(i);  // nothing new for GlobalMergedMesh 
        //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ what uses this for the global mesh ?

        //  THIS IS WALKING OFF THE END OF THE BUFFER FOR mmidx 0 

        idchk->setQuad(nodeinfo.as_vec(), i, 0,0 ); 
        idchk->setQuad(id.as_vec()      , i, 1,0 ); 
        idchk->setQuad(iid.as_vec()     , i, 2,0 ); 

        unsigned MISSING = -1 ; 
        unsigned nface = nodeinfo.x ;
        unsigned nvert = nodeinfo.y ;
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;

        if(i < edgeitems || i % modulo == 0 || i > numVolumes - edgeitems)
        std::cout 
           << " NodeInfo "
           << " nface " << std::setw(6) << nface
           << " nvert " << std::setw(6) << nvert
           << " node " << std::setw(6) << node
           << " parent " << std::setw(6) << ( parent == MISSING ? 0 : parent ) 
           << " Identity "
           << " ( " 
           << std::setw(6) << id.x
           << std::setw(6) << id.y 
           << std::setw(6) << id.z 
           << std::setw(6) << id.w
           << " ) " 
           << " InstancedIdentity "
           << " ( " 
           << std::setw(10) << iid.x
           << " "
           << std::setw(10) << iid.y
           << " "
           << std::setw(10) << iid.z
           << " "
           << std::setw(10) << iid.w
           << " ) " 
           << std::endl 
           ;
    }

    const char* path = SStr::Concat("$TMP/GGeoIdentityTest/",mmidx,"/idchk.npy") ;
    LOG(info) << "write: " << path ; 
    idchk->save(path); 

    const char* cmd = SStr::Concat("a = np.load(os.path.expandvars(\"", path,"\")).reshape(-1,12) ") ; 
    LOG(info) << "np.set_printoptions(edgeitems=10000000, linewidth=200)   " ; 
    LOG(info) << cmd ; 

    return 0 ; 
}
int test_GGeo_identity(const GGeo* gg)
{
    unsigned nmm = gg->getNumMergedMesh(); 
    LOG(info) << " nmm " << nmm ; 
    int rc = 0 ; 
    for(unsigned idx=0 ; idx < nmm ; idx++) rc += test_GGeo_identity(gg, idx);  
    return rc ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);

    GGeo* gg = GGeo::Load(&ok);  

    return test_GGeo_identity(gg);
}


