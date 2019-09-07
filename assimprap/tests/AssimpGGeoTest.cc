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

#include <cassert>

#include "AssimpGGeo.hh"
#include "AssimpTree.hh"
#include "AssimpSelection.hh"
#include "AssimpImporter.hh"

#include "GMesh.hh"
#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GSurfaceLib.hh"
#include "GGeo.hh"

#include "OpticksQuery.hh"
#include "OpticksResource.hh"
#include "Opticks.hh"

#include "OPTICKS_LOG.hh"

// cf canonical int AssimpGGeo::load(GGeo*) 


void test_convertMesh(AssimpGGeo* agg)
{
    unsigned nmesh = agg->getNumMeshes();

    for(unsigned i=0 ; i < nmesh ; i++)
    {
        GMesh* gm = agg->convertMesh(i); 
        gm->Summary();
    }

    GMesh* iav = agg->convertMesh("iav");
    GMesh* oav = agg->convertMesh("oav");

    if(iav) iav->Summary("iav");
    if(oav) oav->Summary("oav");

    LOG(info) << " test_convertMesh DONE " ; 
}


void test_getSensor(GGeo* gg)
{
    GSurfaceLib* m_slib = gg->getSurfaceLib();

    GPropertyMap<float>*  m_sensor_surface = m_slib->getSensorSurface(0) ;

    if(m_sensor_surface == NULL)
    {   
        LOG(warning) << "test_getSensor"
                     << " surface lib sensor_surface NULL "
                     ;   
    }   
    else
    {   
        m_sensor_surface->Summary("test_getSensor  cathode_surface");
    }

}

void test_hasBndLib(GGeo* gg)
{
    GBndLib* blib = gg->getBndLib(); 
    assert( blib ) ; 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);

    ok.configure();

    ok.setGeocache(false);  // prevent loading from any preexisting geocache, same as --nogeocache/-G option

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

    test_hasBndLib(ggeo); //

    LOG(info) << " import " << path ; 

    int importVerbosity = ok.getImportVerbosity();

    AssimpImporter assimp(path, importVerbosity);

    assimp.import();

    LOG(info) << " select " ; 

    AssimpSelection* selection = assimp.select(query);

    AssimpTree* tree = assimp.getTree();

    AssimpGGeo agg(ggeo, tree, selection, query); 

    LOG(info) << " convert " ; 

    int rc = agg.convert(ctrl);

    LOG(info) << " convert DONE " ; 
     
    assert(rc == 0);

    //test_convertMesh(&agg);

    test_getSensor(ggeo);


    return 0 ;
}

