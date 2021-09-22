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
#include "SGeo.hh"
#include "NPY.hpp"
#include "NGLM.hpp"

#include "Opticks.hh"
#include "OpticksIdentity.hh"
#include "OpticksShape.hh"
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




void test_GGeo(const GGeo* gg)
{
    GMergedMesh* mm = gg->getMergedMesh(0);
    unsigned numVolumes = mm->getNumVolumes();
    LOG(info) << " numVolumes " << numVolumes ; 

    GBndLib* blib = gg->getBndLib();

    unsigned UNSET = -1 ; 

    typedef std::pair<std::string, std::string> PSS ; 
    std::set<PSS> pvp ; 

    for(unsigned i=0 ; i < numVolumes ; i++)
    {
        guint4 ni = mm->getNodeInfo(i);
        unsigned nface = ni.x ;
        unsigned nvert = ni.y ;
        unsigned node = ni.z ;
        unsigned parent = ni.w ;
        // assert( node == i ); // no longer true, mm0 now contains just the non-instanced remainder volumes

        glm::uvec4 id = mm->getIdentity_(i);
        unsigned node2 = id.x ;
        unsigned rpo   = id.y ;
        unsigned shape = id.z ; 

        unsigned sensor = id.w ;

        unsigned ridx = OpticksIdentity::RepeatIndex(rpo);  
        unsigned pidx = OpticksIdentity::PlacementIndex(rpo);  
        unsigned oidx = OpticksIdentity::OffsetIndex(rpo);  

        assert( ridx == 0 ); 
        assert( pidx == 0 );
        assert( oidx == i );  
        
        unsigned mesh = OpticksShape::MeshIndex(shape) ;
        unsigned boundary = OpticksShape::BoundaryIndex(shape) ;
        
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

        


        const char* ppv = parent == UNSET ? NULL : gg->getPVName(parent) ;
        const char* pv = gg->getPVName(i) ;
        //const char* lv = gg->getLVName(i) ;

        bool hasSurface =  isur != UNSET || osur != UNSET ; 

        bool select = hasSurface && sensor != UNSET ; 

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
               << std::setw(100) << pv
               << std::setw(100) << ppv
               << std::endl ; 

    }

} 


void test_GGeo_getTransform(const GGeo* gg)
{
    unsigned tot_volumes = gg->getNumVolumes();  
    unsigned num_repeats = gg->getNumRepeats(); 
    LOG(info) 
        << " tot_volumes " << tot_volumes 
        << " num_repeats " << num_repeats 
        ; 

    NPY<float>* transforms0 = gg->getTransforms(); 
    NPY<float>* transforms = NPY<float>::make(tot_volumes, 4, 4); 
    transforms->zero();
    for(unsigned ridx=0 ; ridx < num_repeats ; ridx++)
    {
        unsigned num_placements = gg->getNumPlacements(ridx); 
        unsigned num_volumes = gg->getNumVolumes(ridx); 
        std::cout 
            << " ridx " << std::setw(3) << ridx
            << " num_placements  " << std::setw(6) << num_placements
            << " num_volumes  " << std::setw(6) << num_volumes
            << std::endl 
            ;     

        for(unsigned pidx=0 ; pidx < num_placements ; pidx++)
        {
            for(unsigned oidx=0 ; oidx < num_volumes ; oidx++)
            {
                 glm::uvec4 id = gg->getIdentity(ridx, pidx, oidx); 
                 unsigned nidx = id.x ; 
                 glm::mat4 tr = gg->getTransform(ridx, pidx, oidx); 
                 transforms->setMat4( tr, nidx ); 
            }
        }
    }
 
    bool dump = true ; 
    float eps = 1e-6 ; 
    unsigned mismatch = NPY<float>::compare(transforms0, transforms, eps, dump);  
    LOG(info) << "mismatch " << mismatch ; 
}





void test_GGeo_getIdentity(const GGeo* gg)
{
    unsigned ridx = 5 ; 
    unsigned num_placements = gg->getNumPlacements(ridx); 
    unsigned num_volumes = gg->getNumVolumes(ridx); 
    LOG(info) 
         << " ridx " << ridx 
         << " num_placements " << num_placements
         << " num_volumes " << num_volumes
         ;
             
    for(unsigned pidx = 0 ; pidx < std::min(3u,num_placements) ; pidx++)
    {
        std::cout << " pidx " << pidx << std::endl ; 
        for(unsigned oidx = 0 ; oidx < num_volumes ; oidx++)
        {
            glm::uvec4 tid = gg->getIdentity(ridx, pidx, oidx); 
            unsigned nidx = tid.x ; 
            glm::uvec4 nid = gg->getIdentity(nidx); 

            std::cout 
                << " ridx " << ridx
                << " pidx " << pidx
                << " oidx " << oidx
                << OpticksIdentity::Desc(" tid", tid)
                << OpticksIdentity::Desc(" nid", nid)
                << std::endl
                ;
        }
    }
}

void test_GGeo_getGDMLAuxTargetNodeIndices(const GGeo* gg)
{
    Opticks* ok = gg->getOpticks(); 
    const char* lvname = ok->getGDMLAuxTargetLVName() ;
    std::vector<unsigned> nidxs ; 
    gg->getNodeIndicesForLVName(nidxs, lvname); 
    gg->dumpNodes(nidxs, "test_GGeo_getGDMLAuxTargetNodeIndices"); 
}


void test_GGeo_getFirstNodeIndexForPVNameStarting(const GGeo* gg)
{
    Opticks* ok = gg->getOpticks(); 
    const char* pvname = ok->getPVName() ;
    int nidx = gg->getFirstNodeIndexForPVNameStarting(pvname); 
    const char* pvname2 = gg->getPVName(nidx);

    LOG(info) 
        << " pvname " << pvname
        << " nidx " << nidx
        << " pvname2 " << pvname2
        ;

    if(pvname)
    {
        assert( strcmp(pvname, pvname2) == 0 ); 
    }
}


void test_GGeo_getSignedBoundary(const GGeo* gg)
{
    Opticks* ok = gg->getOpticks(); 
    const char* spec = ok->getBoundary() ;
    int boundary = gg->getSignedBoundary(spec);  // 1-based, signed   0:UNSET 
    unsigned bidx = std::abs(boundary) - 1 ;    // 0-based         
    bool unset =  bidx == GBndLib::UNSET  ;  


    LOG(info) 
        << " spec " << spec
        << " boundary " << boundary 
        << " bidx " << bidx 
        ;  

    GBndLib* blib = gg->getBndLib(); 
    std::string spec2 = unset  ? "unset" : blib->shortname(bidx); 

    LOG(info) 
        << " spec2 " << spec2
        ;  

    if(!unset && spec)
    {
        const char* pspec = spec[0] == '-' ? (spec+1) : spec ;   // handle negated spec
        assert( strcmp(pspec, spec2.c_str()) == 0 );  
    }

} 

void test_getNumMeshes(const GGeo* ggeo)
{
    unsigned num_meshes_ggeo = ggeo->getNumMeshes(); 
    SGeo* sgeo = (SGeo*)ggeo ; 
    unsigned num_meshes_sgeo = sgeo->getNumMeshes(); 

    LOG(info) 
       << " num_meshes_ggeo " << num_meshes_ggeo
       << " num_meshes_sgeo " << num_meshes_sgeo
       ;

    assert( num_meshes_ggeo == num_meshes_sgeo ); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);

    GGeo* gg = GGeo::Load(&ok); 

    //test_GGeo(gg);
    //test_GGeo_getIdentity(gg);
    //test_GGeo_getTransform(gg);
    //test_GGeo_getGDMLAuxTargetNodeIndices(gg);

    //test_GGeo_getFirstNodeIndexForPVNameStarting(gg);
    //test_GGeo_getSignedBoundary(gg);

    test_getNumMeshes(gg);  


    return 0 ;
}


