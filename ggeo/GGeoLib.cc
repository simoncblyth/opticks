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

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "BStr.hh"
#include "BFile.hh"

#include "NPY.hpp"
#include "NSlice.hpp"
#include "NLODConfig.hpp"
#include "GLMFormat.hpp"

#include "Opticks.hh"
#include "OpticksConst.hh"

#include "GBndLib.hh"
#include "GGeoLib.hh"
#include "GMergedMesh.hh"
#include "GParts.hh"
#include "GPts.hh"
#include "GNode.hh"


#include "GGEO_BODY.hh"
#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

const plog::Severity GGeoLib::LEVEL = PLOG::EnvLevel("GGeoLib","DEBUG") ; 


const char* GGeoLib::GMERGEDMESH = "GMergedMesh" ; 
const char* GGeoLib::GPARTS = "GParts" ; 
const char* GGeoLib::GPTS = "GPts" ; 


GBndLib* GGeoLib::getBndLib() const 
{
    return m_bndlib ; 
}

GGeoLib* GGeoLib::Load(Opticks* opticks, GBndLib* bndlib)
{
    GGeoLib* glib = new GGeoLib(opticks, bndlib);
    glib->loadFromCache();
    return glib ; 
}

GGeoLib::GGeoLib(Opticks* ok, GBndLib* bndlib) 
    :
    m_ok(ok),
    m_lodconfig(m_ok->getLODConfig()), 
    m_lod(m_ok->getLOD()), 
    m_bndlib(bndlib),
    m_mesh_version(NULL),
    m_verbosity(m_ok->getVerbosity()),
    m_geolib(this)
{
    assert(bndlib);
}

unsigned GGeoLib::getNumMergedMesh() const 
{
    return m_merged_mesh.size();
}
unsigned GGeoLib::getVerbosity() const 
{
    return m_verbosity ;
}
void GGeoLib::setMeshVersion(const char* mesh_version)
{
    m_mesh_version = mesh_version ? strdup(mesh_version) : NULL ;
}
const char* GGeoLib::getMeshVersion() const 
{
    return m_mesh_version ;
}



GMergedMesh* GGeoLib::getMergedMesh(unsigned index) const   
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end()) return NULL ;

    GMergedMesh* mm = m_merged_mesh.at(index) ;

    return mm ; 
}

void GGeoLib::loadFromCache()
{
    const char* idpath = m_ok->getIdPath() ;
    loadConstituents(idpath);
}


void GGeoLib::save()
{
    const char* idpath = m_ok->getIdPath() ;
    saveConstituents(idpath);
}



bool GGeoLib::HasCacheConstituent(const char* idpath, unsigned ridx) 
{
    const char* sidx = BStr::itoa(ridx) ;
    std::string smmpath = BFile::FormPath(idpath, GMERGEDMESH, sidx );
    std::string sptpath = BFile::FormPath(idpath, GPARTS,      sidx );

    const char* mmpath = smmpath.c_str();
    const char* ptpath = sptpath.c_str();

    return BFile::ExistsDir(mmpath) && BFile::ExistsDir(ptpath) ; 
}


void GGeoLib::removeConstituents(const char* idpath )
{
   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   

        const char* sidx = BStr::itoa(ridx) ;
        std::string smmpath = BFile::FormPath(idpath, GMERGEDMESH, sidx );
        std::string sptpath = BFile::FormPath(idpath, GPARTS,      sidx );

        const char* mmpath = smmpath.c_str();
        const char* ptpath = sptpath.c_str();

        if(BFile::ExistsDir(mmpath))
        {   
            BFile::RemoveDir(mmpath); 
            LOG(debug) << mmpath ; 
        }

        if(BFile::ExistsDir(ptpath))
        {   
            BFile::RemoveDir(ptpath); 
            LOG(debug) << mmpath ; 
        }
   } 
}


/**
GGeoLib::loadConstituents
---------------------------

* loads GMergedMesh, GParts and associates them 

**/

void GGeoLib::loadConstituents(const char* idpath )
{
   LOG(LEVEL) 
       << " mm.reldir " << GMERGEDMESH
       << " gp.reldir " << GPARTS
       << " MAX_MERGED_MESH  " << MAX_MERGED_MESH 
       ; 

   LOG(LEVEL) << idpath ;

   std::stringstream ss ; 

   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        const char* sidx = BStr::itoa(ridx) ;
        std::string smmpath = BFile::FormPath(idpath,  GMERGEDMESH, sidx );
        std::string sptpath = BFile::FormPath(idpath,  GPARTS,      sidx );
        std::string sptspath = BFile::FormPath(idpath, GPTS,        sidx );

        const char* mmpath = smmpath.c_str();
        const char* ptpath = sptpath.c_str();
        const char* ptspath = sptspath.c_str();

        GMergedMesh* mm_ = BFile::ExistsDir(mmpath) ? GMergedMesh::Load( mmpath, ridx, m_mesh_version ) : NULL ; 
        GParts*      parts = BFile::ExistsDir(ptpath) ? GParts::Load( ptpath ) : NULL ; 
        GPts*        pts = BFile::ExistsDir(ptspath) ? GPts::Load( ptspath ) : NULL ; 


        LOG(LEVEL) 
            << " GMergedMesh " << mm_ 
            << " GParts " << parts 
            << " GPts " << pts
            << std::endl 
            << " mmpath " << mmpath
            << std::endl 
            << " ptpath " << ptpath
            << std::endl 
            << " ptspath " << ptspath
            ; 

        if(parts)
        {
            parts->setBndLib(m_bndlib);
            parts->setVerbosity(m_verbosity);
        }

   
        if( mm_ )
        {
            // equivalent for test geometry in GGeoTest::modifyGeometry

            bool lodify = ridx > 0 && m_lod > 0 && m_lodconfig->instanced_lodify_onload > 0 ;  // NB not global 

            GMergedMesh* mm = lodify  ? GMergedMesh::MakeLODComposite(mm_, m_lodconfig->levels ) : mm_ ;         

            mm->setParts(parts);
            mm->setPts(pts);
            
            mm->setGeoCode( OpticksConst::GEOCODE_ANALYTIC ); 
            //mm->setGeoCode( OpticksConst::GEOCODE_TRIANGULATED );

            m_merged_mesh[ridx] = mm ; 
            ss << std::setw(3) << ridx << "," ; 
        }
        else
        {
            if(parts) LOG(fatal) << " parts exists but mm does not " ;
            assert(parts==NULL);
            LOG(debug)
                 << " no mmdir for ridx " << ridx 
                 ;
        }
   }
   LOG(LEVEL) 
             << " loaded "  << m_merged_mesh.size()
             << " ridx (" << ss.str() << ")" 
             ;
}

void GGeoLib::saveConstituents(const char* idpath)
{
    removeConstituents(idpath); // clean old meshes to avoid duplication when repeat counts go down 

    typedef std::map<unsigned int,GMergedMesh*>::const_iterator MUMI ; 
    for(MUMI it=m_merged_mesh.begin() ; it != m_merged_mesh.end() ; it++)
    {
        unsigned int ridx = it->first ; 
        const char* sidx = BStr::itoa(ridx) ;

        GMergedMesh* mm = it->second ; 
        assert(mm->getIndex() == ridx);
        mm->save(idpath, GMERGEDMESH, sidx ); 


        GParts* pt = mm->getParts() ; 
        if(pt)  
        {          
           std::string partsp_ = BFile::FormPath(idpath, GPARTS, sidx );
           const char* partsp = partsp_.c_str();
           pt->save(partsp); 
        }


        GPts* pts = mm->getPts() ; 
        if(pts)  
        {          
           std::string ptsp_ = BFile::FormPath(idpath, GPTS, sidx );
           const char* ptsp = ptsp_.c_str();
           pts->save(ptsp); 
        }

    }
}



GMergedMesh* GGeoLib::makeMergedMesh(unsigned index, const GNode* base, const GNode* root )
{
    LOG(LEVEL) << " mm " << index ;  

    if(m_merged_mesh.find(index) == m_merged_mesh.end())
    {
        m_merged_mesh[index] = GMergedMesh::Create(index, base, root );
    }
    GMergedMesh* mm = m_merged_mesh[index] ;

    LOG(error) << GMergedMesh::Desc( mm ) ;  

    return mm ;
}

void GGeoLib::setMergedMesh(unsigned int index, GMergedMesh* mm)
{
    if(m_merged_mesh.find(index) != m_merged_mesh.end())
        LOG(warning) << "GGeoLib::setMergedMesh REPLACING GMergedMesh "
                     << " index " << index 
                     ;
    m_merged_mesh[index] = mm ; 
}


void GGeoLib::eraseMergedMesh(unsigned int index)
{
    std::map<unsigned int, GMergedMesh*>::iterator it = m_merged_mesh.find(index);
    if(it == m_merged_mesh.end())
    {
        LOG(warning) << "GGeoLib::eraseMergedMesh NO SUCH  "
                     << " index " << index 
                     ;
    }
    else
    {
        m_merged_mesh.erase(it); 
    }
}

void GGeoLib::clear()
{
    m_merged_mesh.clear(); 
}




std::string GGeoLib::desc() const 
{
    std::stringstream ss ; 

    ss << "GGeoLib"
       << " numMergedMesh " << getNumMergedMesh()
       << " ptr " << this
       ; 

    return ss.str();
}


/**
GGeoLib::dump
---------------

This has been updated for mm0 holding remainder volumes (not all volumes as it used to).

**/

void GGeoLib::dump(const char* msg)
{
    LOG(info) << msg << " " << desc() ; 

    GMergedMesh* mm0 = getMergedMesh(0) ; 
    unsigned num_remainder_volumes = mm0 ?  mm0->getNumVolumes() : -1 ; 
    unsigned nmm = getNumMergedMesh();
    unsigned num_instanced_volumes = 0 ; 
    unsigned num_total_faces = 0 ; 
    unsigned num_total_faces_woi = 0 ; 

    for(unsigned i=0 ; i < nmm ; i++)
    {   
        GMergedMesh* mm = getMergedMesh(i); 
        unsigned numVolumes = mm ? mm->getNumVolumes() : -1 ;
        unsigned numITransforms = mm ? mm->getNumITransforms() : -1 ;
        unsigned numFaces = mm ? mm->getNumFaces() : -1 ; 

        std::cout << GMergedMesh::Desc(mm) << std::endl ; 
        num_instanced_volumes += i > 0 ? numITransforms*numVolumes : 0 ;
        num_total_faces += numFaces ;   
        num_total_faces_woi += numITransforms*numFaces ; 

    }
    std::cout
                << " num_remainder_volumes " << num_remainder_volumes 
                << " num_instanced_volumes " << num_instanced_volumes 
                << " num_remainder_volumes + num_instanced_volumes " << num_remainder_volumes + num_instanced_volumes
                << " num_total_faces " << num_total_faces
                << " num_total_faces_woi " << num_total_faces_woi << " (woi:without instancing) " 
                << std::endl
                ;

    for(unsigned i=0 ; i < nmm ; i++)
    {   
        GMergedMesh* mm = getMergedMesh(i); 
        GPts* pts = mm->getPts(); 
        std::cout 
             << std::setw(4) << i 
             << " pts " << ( pts ? "Y" : "N" )
             << " " << ( pts ? pts->brief().c_str() : "" )
             << std::endl
             ;
    }
}

int GGeoLib::checkMergedMeshes() const 
{
    int nmm = getNumMergedMesh();
    int mia = 0 ;

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = getMergedMesh(i);
        if(m_verbosity > 2) 
        std::cout << "GGeoLib::checkMergedMeshes i:" << std::setw(4) << i << " mm? " << (mm ? int(mm->getIndex()) : -1 ) << std::endl ; 
        if(mm == NULL) mia++ ; 
    } 

    if(m_verbosity > 2 || mia != 0)
    LOG(info) << "GGeoLib::checkMergedMeshes" 
              << " nmm " << nmm
              << " mia " << mia
              ;

    return mia ; 
}


/**
GGeoLib::dryrun_convert
-----------------------------

Dry run the non-GPU parts of OGeo::convert to check all is well with the
geometry prior to doing the real thing, which can cause hard crashes giving 
kernel panics.

**/

void GGeoLib::dryrun_convert() 
{
    LOG(info); 
    m_geolib->dump("GGeoLib::dryrun_convert");

    unsigned int nmm = m_geolib->getNumMergedMesh();

    LOG(info) << "[ nmm " << nmm ;

    for(unsigned i=0 ; i < nmm ; i++) 
    {    
        dryrun_convertMergedMesh(i); 
    }    

    LOG(info) << "] nmm " << nmm  ; 

}
void GGeoLib::dryrun_convertMergedMesh(unsigned i)
{
    LOG(LEVEL) << "( " << i  ; 

    GMergedMesh* mm = m_geolib->getMergedMesh(i);

    bool raylod = m_ok->isRayLOD() ;
    if(raylod) LOG(fatal) << " RayLOD enabled " ;
    assert(raylod == false);  

    bool is_null = mm == NULL ;
    bool is_skip = mm->isSkip() ;
    bool is_empty = mm->isEmpty() ;

    if( is_null || is_skip || is_empty )
    {
        LOG(error) << " not converting mesh " << i << " is_null " << is_null << " is_skip " << is_skip << " is_empty " << is_empty ;
        return  ;
    }

    if( i == 0 )   // global non-instanced geometry in slot 0
    {
        dryrun_makeGlobalGeometryGroup(mm);
    }
    else           // repeated geometry
    {
        dryrun_makeRepeatedAssembly(mm) ;
    }
} 

void GGeoLib::dryrun_makeGlobalGeometryGroup(GMergedMesh* mm)
{
    int dbgmm =  m_ok->getDbgMM() ;
    if(dbgmm == 0) mm->dumpVolumesSelected("GGeoLib::dryrun_makeGlobalGeometryGroup [--dbgmm 0] ");
    dryrun_makeOGeometry( mm );
}


void GGeoLib::dryrun_makeRepeatedAssembly(GMergedMesh* mm)
{
    unsigned mmidx = mm->getIndex();
    unsigned imodulo = m_ok->getInstanceModulo( mmidx );

    LOG(LEVEL)
         << " mmidx " << mmidx
         << " imodulo " << imodulo
         ;


    NPY<float>* itransforms = mm->getITransformsBuffer();
    NPY<unsigned int>* ibuf = mm->getInstancedIdentityBuffer();

    NSlice* islice = mm->getInstanceSlice();
    if(!islice) islice = new NSlice(0, itransforms->getNumItems()) ;

    unsigned int numTransforms = islice->count();
    assert(itransforms && numTransforms > 0);

    unsigned int numIdentity = ibuf->getNumItems();

    assert(numIdentity % numTransforms == 0 && "expecting numIdentity to be integer multiple of numTransforms");
    unsigned int numSolids = numIdentity/numTransforms ;

    unsigned count(0);
    if(imodulo == 0u)
    {
        count = islice->count() ;
    }
    else
    {
        for(unsigned int i=islice->low ; i<islice->high ; i+=islice->step) //  CAUTION HEAVY LOOP eg 20k PMTs 
        {
            if( i % imodulo != 0u ) continue ;
            count++ ;
        }
    }

    LOG(LEVEL)
        << " numTransforms " << numTransforms
        << " numIdentity " << numIdentity
        << " numSolids " << numSolids
        << " islice " << islice->description()
        << " count " << count
        ;



}



void GGeoLib::dryrun_makeOGeometry(GMergedMesh* mm)
{
    char ugeocode  = mm->getCurrentGeoCode();
    LOG(LEVEL) << "ugeocode [" << (char)ugeocode << "]" ;

    if(ugeocode == OpticksConst::GEOCODE_TRIANGULATED )
    {
        dryrun_makeTriangulatedGeometry(mm);
    }
    else if(ugeocode == OpticksConst::GEOCODE_ANALYTIC)
    {
        dryrun_makeAnalyticGeometry(mm);
    }
    else if(ugeocode == OpticksConst::GEOCODE_GEOMETRYTRIANGLES)
    {
        dryrun_makeGeometryTriangles(mm);
    }
    else
    {
        LOG(fatal) << "geocode must be triangulated or analytic, not [" << (char)ugeocode  << "]" ;
        assert(0);
    }
}


 
void GGeoLib::dryrun_makeTriangulatedGeometry(GMergedMesh* mm)
{
    unsigned numVolumes = mm->getNumVolumes();
    unsigned numFaces = mm->getNumFaces();
    unsigned numITransforms = mm->getNumITransforms();
            
    GBuffer* id = mm->getAppropriateRepeatedIdentityBuffer();
    GBuffer* vb = mm->getVerticesBuffer() ;
    GBuffer* ib = mm->getIndicesBuffer() ;

    plog::Severity lev = info ; 

    LOG(lev)
        << " mmIndex " << mm->getIndex()
        << " numFaces (PrimitiveCount) " << numFaces
        << " numVolumes " << numVolumes
        << " numITransforms " << numITransforms
        ;

    LOG(lev) << " identityBuffer " << id->desc(); 
    LOG(lev) << " verticesBuffer " << vb->desc(); 
    LOG(lev) << " indicesBuffer  " << ib->desc(); 
}
void GGeoLib::dryrun_makeGeometryTriangles(GMergedMesh* mm)
{
    dryrun_makeTriangulatedGeometry(mm); 
}


void GGeoLib::dryrun_makeAnalyticGeometry(GMergedMesh* mm)
{
    bool dbgmm = m_ok->getDbgMM() == int(mm->getIndex()) ;
    bool dbganalytic = m_ok->hasOpt("dbganalytic") ;

    GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry ");

    if(pts->getPrimBuffer() == NULL)
    {
        LOG(LEVEL) << "( GParts::close " ;
        pts->close();
        LOG(LEVEL) << ") GParts::close " ;
    }
    else
    {
        LOG(LEVEL) << " skip GParts::close " ;
    }
   
    LOG(LEVEL) << "mm " << mm->getIndex()
              << " verbosity: " << m_verbosity
              << ( dbgmm ? " --dbgmm " : " " )
              << ( dbganalytic ? " --dbganalytic " : " " )
              << " pts: " << pts->desc()
              ;

    if(dbgmm)
    {
        LOG(fatal) << "dumping as instructed by : --dbgmm " << m_ok->getDbgMM() ;
        mm->dumpVolumesSelected("GGeoLib::dryrun_makeAnalyticGeometry");
    }


    if(dbganalytic || dbgmm ) pts->fulldump("--dbganalytic/--dbgmm", 10) ;

    NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
    NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
    NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
    NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim

    unsigned numPrim = primBuf->getNumItems();


    NPY<float>* itransforms = mm->getITransformsBuffer(); assert(itransforms && itransforms->hasShape(-1,4,4) ) ;
    unsigned numInstances = itransforms->getNumItems();
    NPY<unsigned>*  idBuf = mm->getInstancedIdentityBuffer();   assert(idBuf);
    LOG(LEVEL)
        << " mmidx " << mm->getIndex()
        << " numInstances " << numInstances
        << " numPrim " << numPrim
        << " idBuf " << idBuf->getShapeString()
        ;

    if( mm->getIndex() > 0 )  // volume level buffers do not honour selection unless using globalinstance
    {
        assert(idBuf->hasShape(numInstances,numPrim,4));
    }



    unsigned numPart = partBuf->getNumItems();
    unsigned numTran = tranBuf->getNumItems();
    unsigned numPlan = planBuf->getNumItems();

    unsigned numVolumes = mm->getNumVolumes();
    unsigned numVolumesSelected = mm->getNumVolumesSelected();

    if( pts->isNodeTree() )
    {
        bool match = numPrim == numVolumes ;
        if(!match)
        {
            LOG(fatal)
                << " NodeTree : MISMATCH (numPrim != numVolumes) "
                << " (this happens when using --csgskiplv) "
                << " numVolumes " << numVolumes
                << " numVolumesSelected " << numVolumesSelected
                << " numPrim " << numPrim
                << " numPart " << numPart
                << " numTran " << numTran
                << " numPlan " << numPlan
                ;
        }
        //assert( match && "NodeTree Sanity check failed " );
        // hmm tgltf-;tgltf-- violates this ?
    }


    //assert( numPrim < 10 );  // expecting small number
    assert( numTran <= numPart ) ;
}





unsigned GGeoLib::getNumRepeats() const 
{
    return getNumMergedMesh(); 
}
unsigned GGeoLib::getNumPlacements(unsigned ridx) const 
{
    unsigned num_repeats = getNumRepeats(); 
    assert( ridx < num_repeats ); 
    GMergedMesh* mm = getMergedMesh(ridx); 
    assert(mm) ; 
    return mm->getNumITransforms(); 
}
unsigned GGeoLib::getNumVolumes(unsigned ridx) const
{
    unsigned num_repeats = getNumRepeats(); 
    assert( ridx < num_repeats ); 
    GMergedMesh* mm = getMergedMesh(ridx); 
    assert(mm) ; 
    return mm->getNumTransforms(); 
}


bool GGeoLib::checkTriplet(unsigned ridx, unsigned pidx, unsigned oidx) const
{
    unsigned num_repeats = getNumRepeats(); 
    unsigned num_placements = getNumPlacements(ridx); 
    unsigned num_volumes = getNumVolumes(ridx); 

    assert( ridx < num_repeats ); 
    assert( pidx < num_placements ); 
    assert( oidx < num_volumes ); 

    return true ; 
}

/**
GGeoLib::getTransform
-----------------------

cf ana/ggeo.py:get_transform(ridx, pidx, oidx) 

**/
glm::mat4 GGeoLib::getTransform(unsigned ridx, unsigned pidx, unsigned oidx) const 
{
    checkTriplet(ridx, pidx, oidx); 

    GMergedMesh* mm = getMergedMesh(ridx); 
    assert(mm) ; 

    NPY<float>* itbuf = mm->getITransformsBuffer(); 

    glm::mat4 placement_transform = itbuf->getMat4(pidx) ;  

    glm::mat4 offset_transform = mm->getTransform_(oidx);  

    glm::mat4 triplet_transform = placement_transform * offset_transform ;  // ORDER SEEMS CORRECT 

    LOG(LEVEL) << gpresent("pTR",placement_transform) ; 
    LOG(LEVEL) << gpresent("oTR",offset_transform) ; 
    LOG(LEVEL) << gpresent("tTR",triplet_transform) ; 

    return triplet_transform ;
}


glm::uvec4 GGeoLib::getIdentity(unsigned ridx, unsigned pidx, unsigned oidx) const
{
    checkTriplet(ridx, pidx, oidx); 

    GMergedMesh* mm = getMergedMesh(ridx); 
    assert(mm) ; 
    NPY<unsigned>* iib = mm->getInstancedIdentityBuffer(); 

    //glm::uvec4 id = iib->getQuad(pidx, oidx, 0 );   see notes/issues/triplet-id-loosing-offset-index-in-NPY.rst
    glm::uvec4 id = iib->getQuad_(pidx, oidx, 0 ); 

    return id ; 
}

 
/**
GGeoLib::getNodeIndex
----------------------

cf ana/ggeo.py:get_node_index(ridx, pidx, oidx)

**/

unsigned GGeoLib::getNodeIndex(unsigned ridx, unsigned pidx, unsigned oidx) const
{
    glm::uvec4 id = getIdentity(ridx, pidx, oidx);     
    return id.x  ; 
}

