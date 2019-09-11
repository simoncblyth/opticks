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

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>

#include "NGLM.hpp"


// opticks-
#include "Opticks.hh"
#include "OpticksQuery.hh"
#include "OpticksResource.hh"

#include "AssimpGGeo.hh"
#include "AssimpTree.hh"
#include "AssimpSelection.hh"
#include "AssimpImporter.hh"

#include "GMesh.hh"
#include "GGeo.hh"


#include "MWrap.hh"
#include "MMesh.hh"

#include "OPTICKS_LOG.hh"


// huh: should this not be in MFixer ?

int fixmesh(Opticks* ok, GMesh* gm)
{
    MWrap<MMesh> ws(new MMesh);
    ws.load(gm);  // asserting in here

    int ncomp = ws.labelConnectedComponentVertices("component"); 
    printf("fixmesh ncomp: %d \n", ncomp);

    if(ncomp != 2) return 1 ; 

    typedef MMesh::VertexHandle VH ; 
    typedef std::map<VH,VH> VHM ;

    MWrap<MMesh> wa(new MMesh);
    MWrap<MMesh> wb(new MMesh);

    VHM s2c_0 ;  
    ws.partialCopyTo(wa.getMesh(), "component", 0, s2c_0);

    VHM s2c_1 ;  
    ws.partialCopyTo(wb.getMesh(), "component", 1, s2c_1);

    ws.dump("ws",0);
    wa.dump("wa",0);
    wb.dump("wb",0);

    ws.dumpBounds("ws.dumpBounds");
    wb.dumpBounds("wb.dumpBounds");
    wa.dumpBounds("wa.dumpBounds");


    wa.write("$TMP/comp%d.off", 0 );
    wb.write("$TMP/comp%d.off", 1 );

    wa.calcFaceCentroids("centroid"); 
    wb.calcFaceCentroids("centroid"); 

    // xyz delta maximum and w: minimal dot product of normals, -0.999 means very nearly back-to-back
    //glm::vec4 delta(10.f, 10.f, 10.f, -0.999 ); 

    OpticksResource* resource = ok->getResource(); 
    glm::vec4 delta = resource->getMeshfixFacePairingCriteria();

    MWrap<MMesh>::labelSpatialPairs( wa.getMesh(), wb.getMesh(), delta, "centroid", "paired");

    wa.deleteFaces("paired");
    wb.deleteFaces("paired");

    wa.collectBoundaryLoop();
    wb.collectBoundaryLoop();

    VHM a2b = MWrap<MMesh>::findBoundaryVertexMap(&wa, &wb );  

    MWrap<MMesh> wdst(new MMesh);

    wdst.createWithWeldedBoundary( &wa, &wb, a2b );

    GMesh* result = wdst.createGMesh(); 
    result->setVersion("_v0");
    result->save( "$TMP", "OpenMeshRapTest/0" );


    return 0 ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);

    ok.configure();

    ok.setGeocacheEnabled(false); 

    const char* path = ok.getDAEPath();

    OpticksResource* resource = ok.getResource();

    OpticksQuery* query = ok.getQuery() ;

    const char* ctrl = resource->getCtrl() ;

    LOG(info)<< "AssimpGGeoTest "
             << " path " << ( path ? path : "NULL" )
             << " query " << ( query ? query->getQueryString() : "NULL" )
             << " ctrl " << ( ctrl ? ctrl : "NULL" )
             ;

    assert(path);
    assert(query);
    assert(ctrl);


    GGeo* ggeo = new GGeo(&ok);

    LOG(info) << " import " << path ;

    int importVerbosity = ok.getImportVerbosity();

    AssimpImporter assimp(path, importVerbosity);

    assimp.import();

    LOG(info) << " select " ;

    AssimpSelection* selection = assimp.select(query);

    AssimpTree* tree = assimp.getTree();

    AssimpGGeo agg(ggeo, tree, selection, query );

    LOG(info) << " convert " ;

    int rc ;

    rc = agg.convert(ctrl);

    LOG(info) << " convert DONE " ;

    assert(rc == 0);


    const char* meshname = "iav" ;

    GMesh* gm = agg.convertMesh(meshname);

    if(!gm)
    {
       LOG(warning) << "failed to find mesh named " << meshname ; 
       return 0 ;
    }  
    else
    {
        gm->Summary(meshname);
    }

    GMesh* gmd = gm->makeDedupedCopy() ;
    gmd->Summary("dedupedCopy needed for fixmesh");

    rc = fixmesh(&ok, gmd );

    LOG(info) << " fixmesh DONE " ;

    return 0;
}


