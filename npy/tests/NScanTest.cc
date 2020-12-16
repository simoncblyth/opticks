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

/*

NScanTest $TMP/tgltf/extras

*/

#include "BStr.hh"
#include "BFile.hh"
#include "BOpticksResource.hh"

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NScan.hpp"
#include "NNode.hpp"
#include "NSceneConfig.hpp"

#include "OPTICKS_LOG.hh"



void test_scan( const std::vector<NCSG*>& trees )
{
    int ntree = trees.size() ; 
    int verbosity = ntree == 1 ? 4 : 0 ; 

    LOG(info) 
          << " NScanTest autoscan trees " 
          << " ntree " << ntree 
          ; 

    const unsigned MESSAGE_NZERO = 1000001 ;  

    typedef std::map<unsigned, unsigned> MUU ; 
    MUU counts ; 

    //float mmstep = 1.f ; 
    float mmstep = 0.1f ; 

    std::vector<NScan*> scans ; 

    for(unsigned i=0 ; i < trees.size() ; i++)
    {
        NCSG* csg = trees[i]; 
        nnode* root = csg->getRoot();

        NScan* scan = new NScan(*root, verbosity);
        unsigned nzero = scan->autoscan(mmstep);
        const std::string& msg = scan->get_message();
        if(!msg.empty()) counts[MESSAGE_NZERO]++ ; 

        scans.push_back(scan);

        counts[nzero]++ ; 
    }


    unsigned total = trees.size() ;
    LOG(info) << " autoscan non-zero counts"
              << " trees " << total 
              << " mmstep "<< mmstep 
               ; 
    for(MUU::const_iterator it=counts.begin() ; it != counts.end() ; it++)
    {
        std::cout 
           << " nzero " << std::setw(4) << it->first   
           << " count " << std::setw(4) << it->second
           << " frac " << float(it->second)/float(total) 
           << std::endl 
           ;
    }
    
    for(MUU::const_iterator it=counts.begin() ; it != counts.end() ; it++)
    {
        unsigned nzero = it->first  ;
        bool expect = nzero == 2 || nzero == 4 ; 
        bool dump = !expect  ; 

        std::cout 
           << std::endl 
           << " nzero " << std::setw(4) << it->first   
           << " count " << std::setw(4) << it->second
           << " frac " << float(it->second)/float(total) 
           << std::endl 
           ;

        if(!dump) continue ; 

        

        for(unsigned i=0 ; i < scans.size() ; i++)
        {
            NScan* scan = scans[i] ; 
            NCSG*  csg = trees[i] ; 
            nnode* root = csg->getRoot();

            bool with_nzero = scan->get_nzero() == nzero ;
            bool with_message = scan->has_message() && nzero == MESSAGE_NZERO ; 
            unsigned nprim = root->get_num_prim() ; 



            if(with_nzero || with_message)
            {
                std::cout 
                     << " i " << std::setw(4) << i 
                     << " nzero " << std::setw(4) << nzero 
                     << " NScanTest " << std::setw(4) << csg->getTreeNameIdx()
                     //<< std::left << std::setw(40) << csg->getTreeDir()  << std::right
                     //<< " treeNameIdx " << csg->getTreeNameIdx()
                     << " soname " << std::setw(40) << csg->get_soname()  
                     << " tag " << std::setw(10) << root->tag()
                     << " nprim " << std::setw(4) << nprim
                     << " typ " << std::setw(20) << root->get_type_mask_string()
                     << " msg " << scan->get_message()
                     << std::endl 
                     ;
            }
        }
    }


}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    BOpticksResource* rsc = BOpticksResource::Get(NULL) ;  // no Opticks at this level 
    const char* basedir = rsc->getDebuggingTreedir(argc, argv);  // uses debugging only IDPATH envvar

    const char* gltfconfig = "csg_bbox_parsurf=1" ;

    int verbosity = 0 ; 

    if(BFile::pathEndsWithInt(basedir))
    {
        std::vector<NCSG*> trees ;
        NCSG* csg = NCSG::Load(basedir, gltfconfig);
        if(csg) trees.push_back(csg);   
        test_scan(trees); 
    }
    else
    {
        //bool checkmaterial = false ;  // TODO: avoid this switch off, by distinguishing the boundary csg.txt from the uri extras csg.txt
        NCSGList* ls = NCSGList::Load(basedir, verbosity ); 
        if(!ls)
        {
            LOG(warning) << " no basedir " << basedir ; 
            return 0 ; 
        }

        const std::vector<NCSG*>& trees = ls->getTrees() ;
        test_scan(trees); 
    }

    return 0 ; 
}







