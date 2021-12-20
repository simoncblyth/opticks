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

// ggv --geolib

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include "Opticks.hh"

#include "GBuffer.hh"
#include "GMergedMesh.hh"
#include "GBndLib.hh"
#include "GGeoLib.hh"

#include "GGEO_BODY.hh"
#include "OPTICKS_LOG.hh"


void test_InstancedMergedMesh(GMergedMesh* mm)
{
    assert(mm->getIndex() > 0);

    //GBuffer* itransforms = mm->getITransformsBuffer();
    NPY<float>* itransforms = mm->getITransformsBuffer();
    unsigned int numITransforms = itransforms ? itransforms->getNumItems() : 0  ;
    printf("numITransforms %u \n", numITransforms  );


    NPY<unsigned int>* ii = mm->getInstancedIdentityBuffer();
    ii->dump();

    unsigned int ni = ii->getShape(0);
    unsigned int nj = ii->getShape(1);
    unsigned int nk = ii->getShape(2);
    assert(nj >= 1 && nk == 4);

    for(unsigned int i=0 ; i < ni ; i++)
    {
        printf("%d\n", i);
        glm::uvec4 q = ii->getQuadU(i, 0) ;
        print(q, "ii"); // _u
    }
}


void test_GlobalMergedMesh(GMergedMesh* mm)
{
    assert(mm->getIndex() == 0);
    //mm->dumpVolumes("test_GlobalMergedMesh");
    unsigned numVolumes = mm->getNumVolumes();
    for(unsigned i=0 ; i < numVolumes ; i++)
    {
        guint4 nodeinfo = mm->getNodeInfo(i);
        unsigned nface = nodeinfo.x ;
        unsigned nvert = nodeinfo.y ;
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;
        assert( node == i );

        guint4 id = mm->getIdentity(i);
        unsigned node2 = id.x ;
        unsigned mesh = id.y ;
        unsigned boundary = id.z ;
        unsigned sensor = id.w ;
        assert( node2 == i );
        
        guint4 iid = mm->getInstancedIdentity(i);  // nothing new for GlobalMergedMesh 
        assert( iid.x == id.x );
        assert( iid.y == id.y );
        assert( iid.z == id.z );
        assert( iid.w == id.w );

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
             << std::endl 
             ;
    }
}




void test_getFaceRepeatedIdentityBuffer(GMergedMesh* mm)
{
    GBuffer* buf = mm->getFaceRepeatedIdentityBuffer();
    assert(buf);
}
void test_getFaceRepeatedInstancedIdentityBuffer(GMergedMesh* mm)
{
    GBuffer* buf = mm->getFaceRepeatedInstancedIdentityBuffer();
    assert(buf);
}


void test_GGeoLib(GGeoLib* geolib)
{
    unsigned nmm = geolib->getNumMergedMesh();
    for(unsigned i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = geolib->getMergedMesh(i);
        if(!mm) return ;  

        if(i == 0)
        {
            //test_GlobalMergedMesh(mm);
            test_getFaceRepeatedIdentityBuffer(mm);
        }
        else
        {
            //test_InstancedMergedMesh(mm);
            test_getFaceRepeatedInstancedIdentityBuffer(mm);
        }
    }
}


void test_getIdentity(GGeoLib* geolib)
{
    unsigned nmm = geolib->getNumMergedMesh() ; 

    for(unsigned ridx = 0 ; ridx < nmm ; ridx++ )
    {  
        unsigned pidx = 0 ;   
        unsigned oidx = 0 ;   

        GMergedMesh* mm = geolib->getMergedMesh(ridx); 
        unsigned numVol = mm->getNumVolumes(); 

        glm::uvec4 id = geolib->getIdentity(ridx, pidx, oidx); 
        unsigned nidx = id.x ; 
        LOG(info)
            << " (" 
            << " ridx " << ridx 
            << " pidx " << pidx 
            << " oidx " << oidx 
            << " )" 
            << " nidx " << nidx 
            << " numVol " << numVol 
            ;    
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();


    bool constituents = true ; 
    GBndLib* bndlib = GBndLib::load(&ok, constituents);

    GGeoLib* geolib = GGeoLib::Load(&ok, bndlib); 
    std::string s = geolib->summary("geolib");
    LOG(info) << std::endl << s ; 

    //test_GGeoLib(geolib);
    test_getIdentity(geolib); 


    return 0 ; 
}


