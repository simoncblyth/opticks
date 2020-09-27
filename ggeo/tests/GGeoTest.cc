/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <set>
#include <string>

#include "SStr.hh"
#include "NPY.hpp"
#include "NGLM.hpp"

#include "Opticks.hh"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"
#include "GMergedMesh.hh"

#include "OPTICKS_LOG.hh"
#include "GGEO_BODY.hh"

void misc(GGeo* m_ggeo)
{
    unsigned int nmm = m_ggeo->getNumMergedMesh();
    for(unsigned int i=0 ; i < nmm ; i++)
    { 
        GMergedMesh* mm = m_ggeo->getMergedMesh(i) ;
        unsigned int numVolumes = mm->getNumVolumes();
        unsigned int numVolumesSelected = mm->getNumVolumesSelected();

        LOG(info) << " i " << i 
                  << " numVolumes " << numVolumes       
                  << " numVolumesSelected " << numVolumesSelected ;      

        for(unsigned int j=0 ; j < numVolumes ; j++)
        {
            gbbox bb = mm->getBBox(j);
            bb.Summary("bb");
        }

        GBuffer* friid = mm->getFaceRepeatedInstancedIdentityBuffer();
        if(friid) friid->save<unsigned int>("$TMP/friid.npy");

        GBuffer* frid = mm->getFaceRepeatedIdentityBuffer();
        if(frid) frid->save<unsigned int>("$TMP/frid.npy");
    }
}



void test_GGeo_identity(const GGeo* gg, unsigned mmidx)
{
    const GMergedMesh* mm = gg->getMergedMesh(mmidx);
    unsigned numVolumes = mm->getNumVolumes();

    NPY<int>* idchk = NPY<int>::make(numVolumes,3,4) ; 
    idchk->zero(); 

    bool global = mmidx == 0 ; 

    unsigned edgeitems = 20 ; 
    unsigned modulo = 500 ; 
    LOG(info) << " mmidx " << mmidx << " numVolumes " << numVolumes << " edgeitems " << edgeitems << " modulo " << modulo  ; 
    for(unsigned i=0 ; i < numVolumes ; i++)
    {
        guint4 nodeinfo = mm->getNodeInfo(i);
        guint4 id = mm->getIdentity(i);

        guint4 iid = mm->getInstancedIdentity(i);  // nothing new for GlobalMergedMesh 
        //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ what uses this for the global mesh ?

        idchk->setQuad(nodeinfo.as_vec(), i, 0,0 ); 
        idchk->setQuad(id.as_vec()      , i, 1,0 ); 
        idchk->setQuad(iid.as_vec()     , i, 2,0 ); 

        unsigned MISSING = -1 ; 
        unsigned nface = nodeinfo.x ;
        unsigned nvert = nodeinfo.y ;
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;

        if(global) assert( node == i );


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

    const char* path = SStr::Concat("$TMP/GGeoTest/test_GGeo_identity/",mmidx,"/idchk.npy") ;
    LOG(info) << "write: " << path ; 
    idchk->save(path); 

    const char* cmd = SStr::Concat("a = np.load(os.path.expandvars(\"", path,"\")).reshape(-1,12) ") ; 
    LOG(info) << "np.set_printoptions(edgeitems=10000000, linewidth=200)   " ; 
    LOG(info) << cmd ; 
}
void test_GGeo_identity(const GGeo* gg)
{
    unsigned nmm = gg->getNumMergedMesh(); 
    LOG(info) << " nmm " << nmm ; 
    for(unsigned idx=0 ; idx < nmm ; idx++) test_GGeo_identity(gg, idx);  
}


void test_GGeo(const GGeo* gg)
{
    GMergedMesh* mm = gg->getMergedMesh(0);
    unsigned numVolumes = mm->getNumVolumes();
    LOG(info) << " numVolumes " << numVolumes ; 

    GBndLib* blib = gg->getBndLib();

    unsigned unset(-1) ; 

    typedef std::pair<std::string, std::string> PSS ; 
    std::set<PSS> pvp ; 

    for(unsigned i=0 ; i < numVolumes ; i++)
    {
        guint4 nodeinfo = mm->getNodeInfo(i);
        guint4 id = mm->getIdentity(i);

        unsigned nface = nodeinfo.x ;
        unsigned nvert = nodeinfo.y ;
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;
        assert( node == i );


        unsigned node2 = id.x ;
        unsigned mesh = id.y ;
        unsigned boundary = id.z ;
        unsigned sensor = id.w ;
        //assert( node2 == i );
        
     /*
        guint4 iid = mm->getInstancedIdentity(i);  // nothing new for GlobalMergedMesh 
        assert( iid.x == id.x );
        assert( iid.y == id.y );
        assert( iid.z == id.z );
        assert( iid.w == id.w );
     */

        std::string bname = blib->shortname(boundary);
        guint4 bnd = blib->getBnd(boundary);

        //unsigned imat = bnd.x ; 
        unsigned isur = bnd.y ; 
        unsigned osur = bnd.z ; 
        //unsigned omat = bnd.w ; 

        const char* ppv = parent == unset ? NULL : gg->getPVName(parent) ;
        const char* pv = gg->getPVName(i) ;
        //const char* lv = gg->getLVName(i) ;

        bool hasSurface =  isur != unset || osur != unset ; 

        bool select = hasSurface && sensor > 0 ; 

        if(select)
        {
             std::cout 
             << " " << std::setw(8) << i 
             << " ni[" 
             << " " << std::setw(6) << nface
             << " " << std::setw(6) << nvert 
             << " " << std::setw(6) << node
             << " " << std::setw(6) << parent
             << " ]"
             << " id[" 
             << " " << std::setw(6) << node2
             << " " << std::setw(6) << mesh
             << " " << std::setw(6) << boundary
             << " " << std::setw(6) << sensor
             << " ]"
             
             << " " << std::setw(50) << bname
             << std::endl 
             ;

/*
            std::cout << "  lv " << lv << std::endl ; 
            std::cout << "  pv " << pv << std::endl ; 
            std::cout << " ppv " << ppv << std::endl ; 
*/
            pvp.insert(PSS(pv,ppv));

      }

    }
    LOG(info) << " pvp.size " << pvp.size() ; 

    for(std::set<PSS>::const_iterator it=pvp.begin() ; it != pvp.end() ; it++ )
    {
        std::string pv = it->first ; 
        std::string ppv = it->second ; 

        std::cout 
               << std::setw(60) << pv
               << std::setw(60) << ppv
               << std::endl ; 

    }

} 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure(); 

    GGeo gg(&ok);
    gg.loadFromCache();
    gg.dumpStats();

    //test_GGeo(&gg);
    test_GGeo_identity(&gg);


    return 0 ;
}


