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


/**
GGeoTest::createPmtInBox
--------------------------

* hmm : suspect this was a dirty hack

**/

GMergedMesh* GGeoTest::createPmtInBox()
{
    assert( m_config->getNumElements() == 1 && "GGeoTest::createPmtInBox expecting single container " );

    GVolume* container = makeVolumeFromConfig(0); 
    const char* spec = m_config->getBoundary(0);
    const char* container_inner_material = m_bndlib->getInnerMaterialName(spec);
    const char* medium = m_ok->getAnalyticPMTMedium();
    assert( strcmp( container_inner_material, medium ) == 0 );


    //GMergedMesh* mmpmt = loadPmtDirty();
    GMergedMesh* mmpmt = m_pmtlib->getPmt() ;
    assert(mmpmt);

    unsigned pmtNumVolumes = mmpmt->getNumVolumes() ; 
    container->setIndex( pmtNumVolumes );   // <-- HMM: MAYBE THIS SHOULD FEED INTO GParts::setNodeIndex ?

    LOG(info) 
        << " spec " << spec 
        << " container_inner_material " << container_inner_material
        << " pmtNumVolumes " << pmtNumVolumes
        ; 

    GMesh* mesh = const_cast<GMesh*>(container->getMesh()); // TODO: reorg to avoid 
    mesh->setIndex(1000);
    
    GParts* cpts = container->getParts() ;

    cpts->setPrimFlag(CSG_FLAGPARTLIST);  // PmtInBox uses old partlist, not the default CSG_FLAGNODETREE
    cpts->setAnalyticVersion(mmpmt->getParts()->getAnalyticVersion()); // follow the PMT version for the box
    cpts->setNodeIndex(0, pmtNumVolumes);   // NodeIndex used to associate parts to their prim, fixed 5-4-2-1-1 issue yielding 4-4-2-1-1-1

    GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, container, m_verbosity );   

    // hmm this is putting the container at the end... does that matter ?

    if(m_verbosity > 1)
    triangulated->dumpVolumes("GGeoTest::createPmtInBox GMergedMesh::dumpVolumes combined (triangulated) ");

    // needed by OGeo::makeAnalyticGeometry
    NPY<unsigned int>* idBuf = mmpmt->getAnalyticInstancedIdentityBuffer();
    NPY<float>* itransforms = mmpmt->getITransformsBuffer();

    assert(idBuf);
    assert(itransforms);

    triangulated->setAnalyticInstancedIdentityBuffer(idBuf);
    triangulated->setITransformsBuffer(itransforms);

    return triangulated ; 
}


void GGeoTest::createBoxInBox()
{
    unsigned nelem = m_config->getNumElements();
    for(unsigned i=0 ; i < nelem ; i++)
    {
        GVolume* volume = makeVolumeFromConfig(i);
        m_nodelib->add(volume);
    }
}


/**
GGeoTest::makeVolumeFromConfig
---------------------------------

* partlist geometry from commandline strings

**/

GVolume* GGeoTest::makeVolumeFromConfig( unsigned i ) // setup nodeIndex here ?
{
    std::string node = m_config->getNodeString(i);
    OpticksCSG_t type = m_config->getTypeCode(i);

    const char* spec = m_config->getBoundary(i);
    glm::vec4 param = m_config->getParameters(i);
    glm::mat4 trans = m_config->getTransform(i);
    unsigned boundary = m_bndlib->addBoundary(spec);

    LOG(info) 
        << " i " << std::setw(2) << i 
        << " node " << std::setw(20) << node
        << " type " << std::setw(2) << type 
        << " csgName " << std::setw(15) << CSGName(type)
        << " spec " << spec
        << " boundary " << boundary
        << " param " << gformat(param)
        << " trans " << gformat(trans)
        ;

    bool oktype = type < CSG_UNDEFINED ;  
    if(!oktype) LOG(fatal) << "GGeoTest::makeVolumeFromConfig configured node not implemented " << node ;
    assert(oktype);

    GVolume* volume = m_maker->make(i, type, param, spec );   
    GParts* pts = volume->getParts();
    assert(pts);
    pts->setPartList(); // setting primFlag to CSG_FLAGPARTLIST
    pts->setBndLib(m_bndlib) ; 

    return volume ; 
}


/**
GGeoTest::labelPartList
---------------------------

PartList geometry is implemented by allowing a single "primitive" to be composed of multiple
"parts", the association from part to prim being 
controlled via the primIdx attribute of each part.

collected pts are converted into primitives in GParts::makePrimBuffer

NB Partlist test geometry was created 
   via simple simple commandline strings. 
   It is the precursor to proper CSG Trees, which are implemented 
   with NCSG and created using python opticks.analytic.csg.CSG.

**/

void GGeoTest::labelPartList()
{
    for(unsigned i=0 ; i < m_nodelib->getNumVolumes() ; i++)
    {
        GVolume* volume = m_nodelib->getVolume(i) ;
        GParts* pts = volume->getParts();
        assert(pts);
        assert(pts->isPartList());

        OpticksCSG_t csgflag = volume->getCSGFlag(); 
        int flags = csgflag ;

        pts->setIndex(0u, i);
        pts->setNodeIndex(0u, 0 );  
        //
        // for CSG_FLAGPARTLIST the nodeIndex is crucially used to associate parts to their prim 
        // setting all to zero is structuring all parts into a single prim ... 
        // can get away with that for BoxInBox (for now)
        // but would definitely not work for PmtInBox 
        //

        pts->setTypeCode(0u, flags);

        pts->setBndLib(m_bndlib);

        LOG(info) << "GGeoTest::labelPartList"
                  << " i " << std::setw(3) << i 
                  << " csgflag " << std::setw(5) << csgflag 
                  << std::setw(20) << CSGName(csgflag)
                  << " pts " << pts 
                  ;
    }
}


/**
GGeoTest::initCreateBIB
------------------------

This is almost ready to be deleted, once 
succeed to wheel in standard solids
via some proxying metadata in the csglist.

**/

GMergedMesh* GGeoTest::initCreateBIB()
{
    const char* mode = m_config->getMode();
    unsigned nelem = m_config->getNumElements();
    if(nelem == 0 )
    {
        LOG(fatal) << " NULL csgpath and config nelem zero  " ; 
        m_config->dump("GGeoTest::initCreateBIB ERROR nelem==0 " ); 
    }
    assert(nelem > 0);

    GMergedMesh* tmm = NULL ;

    if(m_config->isBoxInBox()) 
    {
        createBoxInBox(); 
        labelPartList() ;
        tmm = combineVolumes(NULL);
    }
    else if(m_config->isPmtInBox())
    {
        assert(0); 
    }
    else 
    { 
        LOG(fatal) << "mode not recognized [" << mode << "]" ; 
        assert(0);
    }

    return tmm ; 
}



