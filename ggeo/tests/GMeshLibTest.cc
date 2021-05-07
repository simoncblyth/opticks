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

// TEST=GMeshLibTest om-t

/**
GMeshLibTest.cc
==================

Loads the GMeshLib include NCSG solids from geocache and
allows dumping.

In direct workflow only need one envvar, the OPTICKS_KEY to identify the 
geocache to load from and the "--envkey" option to switch on sensitivity to it::
 
   GMeshLibTest --envkey --dbgmesh sFasteners
 

In legacy workflow needed the op.sh script to set multiple envvars in order to 
find the geocache::
 
   op.sh --dsst --gmeshlib --dbgmesh near_top_cover_box0xc23f970  
   op.sh --dsst --gmeshlib --dbgmesh near_top_cover_box0x


dsst
    sets geometry selection envvars, defining the path to the geocache
gmeshlib
    used by op.sh script to pick this executable GMeshLibTest 
dbgmesh
    name of mesh to dump 



::

    epsilon:ggeo blyth$ GMeshLibTest $(seq 98 109)
    NNVTMCPPMTsMask0x3c9fa80
    NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x3503950
    NNVTMCPPMT_PMT_20inch_inner2_solid0x3cae8f0
    NNVTMCPPMT_PMT_20inch_body_solid0x3cad240
    NNVTMCPPMT_PMT_20inch_pmt_solid0x3ca9320
    NNVTMCPPMTsMask_virtual0x3cb3b40
    HamamatsuR12860sMask0x3c9afa0
    HamamatsuR12860_PMT_20inch_inner1_solid_I0x3c96fa0
    HamamatsuR12860_PMT_20inch_inner2_solid_1_90x3c93610
    HamamatsuR12860_PMT_20inch_body_solid_1_90x3ca7680
    HamamatsuR12860_PMT_20inch_pmt_solid_1_90x3cb68e0
    HamamatsuR12860sMask_virtual0x3c99fb0


    epsilon:ggeo blyth$ GMeshLibTest NNVTMCPPMTsMask0x NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x NNVTMCPPMT_PMT_20inch_inner2_solid0x
    98
    99
    100


**/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "Opticks.hh"
#include "NCSG.hpp"
#include "GMesh.hh"
#include "GMeshLib.hh"
#include "GMesh.hh"
#include "NBBox.hpp"
#include "NQuad.hpp"
#include "NNode.hpp"
#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

#include "OPTICKS_LOG.hh"

const plog::Severity LEVEL = info ; 


void test_getDbgMeshByName(const Opticks& ok, const GMeshLib* meshlib)
{
    const char* dbgmesh = ok.getDbgMesh();
    if(dbgmesh)
    {
        bool startswith = true ; 
        const GMesh* mesh = meshlib->getMeshWithName(dbgmesh, startswith);
        mesh->dump("GMesh::dump", 50);
        const NCSG* solid = mesh->getCSG(); 
        assert( solid );     
        solid->dump();  
    }
    else
    {
        LOG(info) << "no dbgmesh" ; 
    }
}


void test_dump0( const GMeshLib* meshlib )
{
    unsigned num_mesh = meshlib->getNumMeshes(); 
    LOG(info) << " num_mesh " << num_mesh ; 

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        const GMesh* mesh = meshlib->getMeshSimple(i); 
        const char* name = mesh->getName() ; 

        const NCSG* solid = mesh->getCSG(); 
        nbbox bba = solid->bbox(); // global frame bbox
        nvec4 ce = bba.center_extent() ; 

        //nnode* root = solid->getRoot(); 
        //if( root->transform && !root->transform->is_identity() ) LOG(info) << " tr " << *root->transform ; 

        std::cout  
            << std::setw(2) << i 
            << std::setw(45) << ( name ? name : "NULL" )
            << " bba " << bba.description()
            << " ce " << std::setw(25) << ce.desc()
            << " " << std::setw(2) << i 
            << std::endl    
            ; 
    }
}


void test_dump1( const GMeshLib* meshlib )
{
    unsigned num_mesh = meshlib->getNumMeshes(); 
    LOG(info) << " num_mesh " << num_mesh ; 

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        const GMesh* mesh = meshlib->getMeshSimple(i); 
        const char* name = mesh->getName() ; 
        const NCSG* csg = mesh->getCSG(); 
        glm::vec4 ce0 = csg->bbox_center_extent(); 

         
        //const_cast<NCSG*>(csg)->apply_centering(); 
        const_cast<GMesh*>(mesh)->applyCentering(); 

        int w(40);   

        glm::vec4 ce1 = csg->bbox_center_extent(); 
        std::cout  
            << std::setw(2) << i 
            << std::setw(45) << ( name ? name : "NULL" )
            << " ce0 " << std::setw(w) << gformat(ce0)
            << " ce1 " << std::setw(w) << gformat(ce1)
            << " " << std::setw(2) << i 
            << std::endl    
            ; 
    }
}

void test_getMeshIndexWithName( const GMeshLib* mlib, int argc, char** argv )
{
    LOG(info) ; 
    for(int i=1 ; i < argc ; i++ )
    {
        const char* arg = argv[i] ; 
        if( arg[0] == '-' ) continue ; 

        bool startswith = true ; 
        int midx =  mlib->getMeshIndexWithName( arg, startswith ); 

        LOG(info) 
            << " i " << std::setw(3) << i 
            << " arg " << std::setw(50) << arg 
            << " midx " << std::setw(5) << midx 
            ;

        std::cerr << midx << std::endl ; 
    }
}

void test_getMeshName( const GMeshLib* mlib, int argc, char** argv )
{
    LOG(info) ; 
    for(int i=1 ; i < argc ; i++ )
    {
        const char* arg = argv[i] ; 
        if( arg[0] == '-' ) continue ; 

        int midx =  std::atoi(arg) ; 
        const char* soname = mlib->getMeshName(midx);  
        LOG(info) 
            << " i " << std::setw(3) << i 
            << " midx " << std::setw(5) << midx 
            << " soname " << std::setw(50) << soname 
            ;

        std::cerr << soname << std::endl ; 
    }
}

void test_getMeshName_getMeshIndexWithName( const GMeshLib* mlib, int argc, char** argv )
{
    //LOG(LEVEL) ; 

    bool dump = false ;  
    const char* dump_arg = "--dump" ; 

    for(int i=1 ; i < argc ; i++ )
    {
        const char* arg = argv[i] ; 
        if( strlen(arg) >= strlen(dump_arg) && strcmp(arg, dump_arg) == 0 ) dump = true ; 
        if( strlen(arg) > 2 && arg[0] == '-' && arg[1] == '-' ) continue ; 

        char* end ;  
        char** endptr = &end ; 
        int base = 10 ;  
        unsigned long int uli = strtoul(arg, endptr, base); 
        bool end_points_to_terminator = end == arg + strlen(arg) ;  
        
        if(dump) LOG(LEVEL)
            << " i " << std::setw(3) << i 
            << " arg " << std::setw(50) << arg
            << " uli " << std::setw(10) << uli 
            << " end " << end 
            << " end_points_to_terminator " << ( end_points_to_terminator  ? "Y" : "N" ) 
            ;

        if( end_points_to_terminator )  // succeeded to parse entire string as an integer
        {
            const char* soname = mlib->getMeshName(uli);  
            if(dump) 
            {
                std::cerr
                    << std::setw(10) << uli 
                    << std::setw(50) << ( soname ? soname : "FAIL" ) 
                    << std::endl 
                    ;
            }
            else
            {
                std::cerr << ( soname ? soname : "FAIL" ) << std::endl ; 
            }
        }
        else
        {
            bool startswith = true ; 
            int midx =  mlib->getMeshIndexWithName( arg, startswith ); 
            if(dump)
            {
                std::cerr 
                    << std::setw(50) << arg
                    << std::setw(10) << midx 
                    << std::endl 
                    ;
            }
            else
            {
                std::cerr << midx << std::endl ;
            }
        }
    }
}


void test_operator( const GMeshLib* mlib, int argc, char** argv )
{
    const char* dump_arg = "--dump" ; 
    bool dump = false ; 
    for(int i=1 ; i < argc ; i++ )
    {
        const char* arg = argv[i] ; 
        if( strlen(arg) >= strlen(dump_arg) && strcmp(arg, dump_arg) == 0 ) dump = true ; 
        if( strlen(arg) > 2 && arg[0] == '-' && arg[1] == '-' ) continue ; 
 
        std::string s = (*mlib)(arg) ; 
        std::cerr << s << std::endl ;  
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);  // --envkey and direct geometry is now the default 
    ok.configure();
    if(!ok.isDirect())
    {
        LOG(fatal) << "this is a direct only test : that means use --envkey option and have a valid OPTICKS_KEY envvar "  ; 
        return 0 ; 
    }


    GMeshLib* mlib = GMeshLib::Load(&ok);

    //test_getDbgMeshByName( ok, mlib ); 
    //test_dump0( mlib ); 
    //test_dump1( mlib ); 

    //test_getMeshIndexWithName( meshlib, argc, argv );  
    //test_getMeshName( mlib, argc, argv );  

    //test_getMeshName_getMeshIndexWithName( mlib, argc, argv ); 
    test_operator( mlib, argc, argv ); 

    return 0 ; 
}

