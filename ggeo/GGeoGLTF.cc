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

#include "BFile.hh"

#include "GGeo.hh"
#include "GMesh.hh"
#include "GVolume.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"
#include "GGeoGLTF.hh"

#include "NNode.hpp"
#include "NCSG.hpp"

#include "YOG.hh"
#include "YOGMaker.hh"

#include "PLOG.hh"

using YOG::Sc ; 
using YOG::Nd ; 
using YOG::Mh ; 
using YOG::Maker ; 


const plog::Severity GGeoGLTF::LEVEL = debug ; 


void GGeoGLTF::Save( const GGeo* ggeo, const char* path, int root ) // static
{
    GGeoGLTF tf(ggeo); 
    tf.save(path, root); 

}

GGeoGLTF::GGeoGLTF( const GGeo* ggeo )
    :
    m_ggeo(ggeo),
    m_mlib(ggeo->getMaterialLib()),
    m_blib(ggeo->getBndLib()),
    m_sc(new YOG::Sc(0)),
    m_maker(NULL)
{
    init();
}

void GGeoGLTF::init()
{
    addMaterials();
    addMeshes();
    addNodes();
}

void GGeoGLTF::addMaterials()
{
    unsigned num_materials = m_mlib->getNumMaterials(); 

    for(size_t i=0 ; i < num_materials ; i++)
    {
        const char* name = m_mlib->getName(i);
        int idx = m_sc->add_material(name);
        assert( idx == int(i) );
    }
}

void GGeoGLTF::addMeshes()
{
    unsigned num_meshes = m_ggeo->getNumMeshes();

    for(unsigned i=0 ; i < num_meshes ; i++)
    {
        int lvIdx = i ; 
        const GMesh* mesh = m_ggeo->getMesh(lvIdx); 
        const NCSG* csg = mesh->getCSG(); 
        const nnode* root = mesh->getRoot();  
        const nnode* raw = root->other ;

        std::string lvname = csg->get_lvname();  // <-- probably wont work postcache : IT DOES NOW VIA METADATA
        std::string soname = csg->get_soname(); 

        LOG(verbose)
            << " lvIdx " << lvIdx 
            << " lvname " << lvname 
            << " soname " << soname 
            ;

        int soIdx = m_sc->add_mesh( lvIdx, lvname.c_str(), soname.c_str() );
        assert( soIdx == int(lvIdx) ); 

        Mh* mh = m_sc->meshes[lvIdx] ;

        mh->mesh = mesh ; 
        mh->csg = csg ; 
        mh->csgnode = root ; 
        mh->vtx = mesh->m_x4src_vtx ; 
        mh->idx = mesh->m_x4src_idx ; 

        GSolidRec rec(raw, root, csg, soIdx, lvIdx );
        m_solidrec.push_back( rec ) ; 
    }
}

void GGeoGLTF::addNodes()
{
    const GVolume* top = m_ggeo->getVolume(0); 
    Nd* parent_nd = NULL ; 

    addNodes_r( top, parent_nd, 0 ) ; 
}


void GGeoGLTF::addNodes_r(const GVolume* volume, YOG::Nd* parent_nd, int depth)
{
    const GMesh* mesh = volume->getMesh();  
    int lvIdx = mesh->getIndex() ;   

    const nmat4triple* ltriple = volume->getLocalTransform(); 
    unsigned boundary = volume->getBoundary(); 
    std::string boundaryName = m_blib->shortname(boundary);
    int materialIdx = m_blib->getInnerMaterial(boundary);
    const char* pvName = volume->getPVName() ; 

    LOG(verbose)
         << " volume " << volume 
         << " lv " << lvIdx 
         << " boundary " << std::setw(4) << boundary
         << " materialIdx " << std::setw(4) << materialIdx
         << " boundaryName " << boundaryName
         ;

    int ndIdx = m_sc->add_node(
                               lvIdx,
                               materialIdx,
                               pvName,
                               ltriple,
                               boundaryName,
                               depth,
                               true,      // selected: not yet used in YOG machinery  
                               parent_nd
                               );

    Nd* nd = m_sc->get_node(ndIdx) ; 

    for(unsigned i = 0; i < volume->getNumChildren(); i++) addNodes_r(volume->getChildVolume(i), nd, depth + 1);
}


void GGeoGLTF::save(const char* path, int root )  
{
    m_sc->root = root ;

    LOG(info) 
              << " path " << path 
              << " sc.root " << m_sc->root
              ;

    bool yzFlip = true ;
    bool saveNPYToGLTF = false ;

    BFile::preparePath( path ) ; 

    m_maker = new YOG::Maker(m_sc, yzFlip, saveNPYToGLTF) ;
    m_maker->convert();
    m_maker->save(path);

    std::string dir = BFile::ParentDir(path); 
    writeSolidRec(dir.c_str()); 
}


void GGeoGLTF::dumpSolidRec(const char* msg) const 
{
    LOG(error) << msg ; 
    std::ostream& out = std::cout ;
    solidRecTable( out );  
}

void GGeoGLTF::writeSolidRec(const char* dir) const 
{
    std::string path = BFile::preparePath( dir, "solids.txt", true ) ; 
    LOG(LEVEL) << " writeSolidRec " 
               << " dir [" << dir << "]" 
               << " path [" << path << "]" ;   
    std::ofstream out(path.c_str());
    solidRecTable( out );  
}

void GGeoGLTF::solidRecTable( std::ostream& out ) const 
{
    unsigned num_solid = m_solidrec.size() ; 
    out << "written by GGeoGLTF::solidRecTable " << std::endl ; 
    out << "num_solid " << num_solid << std::endl ; 
    for(unsigned i=0 ; i < num_solid ; i++)
    {   
        const GSolidRec& rec = m_solidrec[i] ; 
        out << rec.desc() << std::endl ; 
    }   
}

