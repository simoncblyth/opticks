#include <csignal>
#include <ostream>
#include <fstream>
#include <iomanip>

#include "BStr.hh"
#include "BFile.hh"
#include "Opticks.hh"

#include "NCSG.hpp"

#include "GMesh.hh"
#include "GMeshLib.hh"
#include "GItemIndex.hh"   // <-- aim to remove 
#include "GItemList.hh"

#include "PLOG.hh"

const plog::Severity GMeshLib::LEVEL = debug ; 


const unsigned GMeshLib::MAX_MESH = 250 ;   // <-- hmm 500 too large ? it means a lot of filesystem checking 


const char* GMeshLib::GITEMINDEX = "GItemIndex" ; 
const char* GMeshLib::GMESHLIB_INDEX = "MeshIndex" ; 

const char* GMeshLib::GMESHLIB_LIST = "GMeshLib.txt" ; 


const char* GMeshLib::GMESHLIB = "GMeshLib" ; 
const char* GMeshLib::GMESHLIB_NCSG = "GMeshLibNCSG" ; 


GMeshLib::GMeshLib(Opticks* ok ) 
   :
   m_ok(ok),
   m_reldir(GMESHLIB),
   m_reldir_solids(GMESHLIB_NCSG),
   m_meshindex(NULL),
   m_meshnames(NULL),
   m_missing(std::numeric_limits<unsigned>::max())
{
}

GMeshLib* GMeshLib::Load(Opticks* ok)
{
    GMeshLib* meshlib = new GMeshLib(ok);
    meshlib->loadFromCache(); 
    return meshlib ; 
}

void GMeshLib::loadFromCache()
{
    const char* idpath = m_ok->getIdPath() ;


    m_meshindex = GItemIndex::load(idpath, GITEMINDEX, GMESHLIB_INDEX ) ;
    assert(m_meshindex);
    bool has_index = m_meshindex->hasIndex() ;
    if(!has_index)  LOG(fatal) << " meshindex load failure " ; 
    assert(has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G ");


    m_meshnames = GItemList::Load(idpath, "GItemList", GMESHLIB_LIST ) ;
    assert(m_meshnames);

    loadMeshes(idpath);
}

void GMeshLib::save() const
{
    const char* idpath = m_ok->getIdPath() ;


    assert( m_meshindex ); 
    m_meshindex->save(idpath);


    assert( m_meshnames ); 
    m_meshnames->save(idpath, "GItemList", GMESHLIB_LIST );

    saveMeshes(idpath);
    saveMeshUsage(idpath);
}



GItemIndex* GMeshLib::getMeshIndex()
{
    return m_meshindex ; 
}
const char* GMeshLib::getMeshName(unsigned aindex) 
{
    return m_meshindex->getNameSource(aindex);
}
unsigned GMeshLib::getMeshIndex(const char* name, bool startswith)  const 
{
    unsigned aindex = startswith ? 
          m_meshindex->getIndexSourceStarting(name, m_missing)
          :
          m_meshindex->getIndexSource(name, m_missing)
          ;
 
    assert( aindex != m_missing );
    return aindex ; 
}

void GMeshLib::dump(const char* msg) const
{
    unsigned num_mesh = m_meshindex->getNumItems();
    LOG(info) << msg 
              << " num_mesh " << num_mesh 
              ; 

    //m_meshindex->dump();

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        const char* name = m_meshindex->getNameSource(i);  // hmm assumes source index is 0:N-1 which is not gauranteed
        //const char* name = m_meshindex->getNameLocal(i);

        assert(name);
        unsigned aindex = m_meshindex->getIndexSource(name, m_missing); // huh : why have to go via name
        assert(aindex != m_missing);
        assert(aindex == i);

        std::cout 
               << " aidx " << std::setw(3) << aindex  
               << " name " << std::setw(50) << name 
               ;

         const GMesh* mesh = getMesh(aindex);
         if(mesh)
         {
             assert( strcmp(mesh->getName(), name) == 0);
             assert( mesh->getIndex() == aindex );
             std::cout << mesh->desc() << std::endl ; 
         }
         else
         {
             std::cout << " NO MESH "  << std::endl ; 
         }
    }
}







unsigned GMeshLib::getNumMeshes() const 
{
    return m_meshes.size();
}
unsigned GMeshLib::getNumSolids() const 
{
    return m_solids.size();
}

const GMesh* GMeshLib::getMesh(unsigned aindex) const 
{
    const GMesh* mesh = NULL ; 
    for(unsigned i=0 ; i < m_meshes.size() ; i++ )
    { 
        if(m_meshes[i]->getIndex() == aindex )
        {
            mesh = m_meshes[i] ; 
            break ; 
        }
    }
    return mesh ;
}  


const NCSG* GMeshLib::getSolid(unsigned aindex) const 
{
    const NCSG* solid = NULL ; 
    for(unsigned i=0 ; i < m_solids.size() ; i++ )
    { 
        if(m_solids[i]->getIndex() == aindex )
        {
            solid = m_solids[i] ; 
            break ; 
        }
    }
    return solid ;
}  





const GMesh* GMeshLib::getMesh(const char* name, bool startswith) const 
{
    unsigned aindex = getMeshIndex(name, startswith);
    return getMesh(aindex);
}


/**
GMeshLib::add
----------------

Invoked via GGeo::add from X4PhysicalVolume::convertSolids_r as each distinct 
solid is encountered in the recursive traverse.

**/

void GMeshLib::add(const GMesh* mesh)
{
    if(!m_meshindex) m_meshindex = new GItemIndex(GITEMINDEX, GMESHLIB_INDEX )   ;
    if(!m_meshnames) m_meshnames = new GItemList("GMeshLib", "GItemList" )   ;

    const char* name = mesh->getName();
    unsigned int index = mesh->getIndex();
    assert(name) ; 

    m_meshnames->add(name); 

    LOG(debug) 
        << " index " << std::setw(4) << index 
        << " name " << name 
        ;

    m_meshes.push_back(mesh);
    const NCSG* solid = mesh->getCSG(); 

    assert(solid) ;                // hmm probably fail for legacy workflow
    m_solids.push_back(solid); 

    m_meshindex->add(name, index); 

    //std::raise(SIGINT); 
}


void GMeshLib::removeDirs(const char* idpath ) const 
{
   for(unsigned int idx=0 ; idx < MAX_MESH ; ++idx)
   {   
        const char* sidx = BStr::itoa(idx);
        if(BFile::ExistsDir(idpath, m_reldir, sidx))
        { 
            BFile::RemoveDir(idpath, m_reldir, sidx); 
        }
        if(BFile::ExistsDir(idpath, m_reldir_solids, sidx))
        { 
            BFile::RemoveDir(idpath, m_reldir_solids, sidx); 
        }
   } 
}

/**
GMeshLib::loadMeshes
----------------------

In addition to GMesh instances this also loads the corresponding NCSG 
analytic solids and associates them with the GMesh instances.

**/

void GMeshLib::loadMeshes(const char* idpath )
{
   LOG(LEVEL) << "idpath "  << idpath ;  

   // TODO: read the directory instead of just checking existance of MAX_MESH paths ?
   //       (or use the index ?)

   for(unsigned int idx=0 ; idx < MAX_MESH ; ++idx)
   {   
        const char* sidx = BStr::itoa(idx);

        std::string meshdir_ = BFile::FormPath(idpath, m_reldir, sidx);
        const char* meshdir = meshdir_.c_str() ;

        std::string soliddir_ = BFile::FormPath(idpath, m_reldir_solids, sidx);
        const char* soliddir = soliddir_.c_str() ;

        bool meshdir_exists = BFile::ExistsDir(meshdir) ; 
        bool soliddir_exists = BFile::ExistsDir(soliddir) ; 

        if(meshdir_exists)
        {   
            assert( soliddir_exists && "GMeshLib persisted GMesh are expected to have paired GMeshLibNCSG dirs"); 

            LOG(debug) 
                << " meshdir " << meshdir 
                << " meshdir_exists " << meshdir_exists 
                << " soliddir " << soliddir 
                << " soliddir_exists " << soliddir_exists 
                ; 

            GMesh* mesh = GMesh::load( meshdir );
            NCSG* solid = NCSG::Load( soliddir );  
            mesh->setCSG(solid); 

            const char* name = getMeshName(idx);
            assert(name);

            // mesh->updateBounds(); // CAN THIS BE DONE IN LOAD ? Perhaps MM complications
           // should have been done by GMesh::loadBuffers ?

            mesh->setIndex(idx);
            mesh->setName(strdup(name));

            m_meshes.push_back(mesh);
            m_solids.push_back(solid);
        }
        else
        {
            LOG(debug)
                << " MISSING meshdir "  
                << " idx " << idx 
                << " meshdir " << meshdir 
                ;
        }
   }
   LOG(info) 
       << " loaded "  
       << " meshes "  << m_meshes.size()
       << " solids "  << m_solids.size()
       ;
}

void GMeshLib::saveMeshes(const char* idpath) const 
{
    removeDirs(idpath); // clean old meshes to avoid duplication when repeat counts go down 

    typedef std::vector<const GMesh*>::const_iterator VMI ; 
    for(VMI it=m_meshes.begin() ; it != m_meshes.end() ; it++)
    {
        const GMesh* mesh = *it ; 
        unsigned int idx = mesh->getIndex() ; 
        const char* sidx = BStr::itoa(idx);

        mesh->save(idpath, m_reldir, sidx); 

        const NCSG* solid = mesh->getCSG(); 
        assert(solid); 

        solid->savesrc(idpath, m_reldir_solids, sidx); 
    }

    // meshindex persisted first, up in GMeshLib::save
}



void GMeshLib::countMeshUsage(unsigned int meshIndex, unsigned int nodeIndex)
{
     // called during GGeo creation from: void AssimpGGeo::convertStructure(GGeo* gg)
     //printf("GMeshLib::countMeshUsage %d %d %s %s \n", meshIndex, nodeIndex, lv, pv);
     m_mesh_usage[meshIndex] += 1 ; 
     m_mesh_nodes[meshIndex].push_back(nodeIndex); 
}


std::map<unsigned int, unsigned int>& GMeshLib::getMeshUsage()
{
    return m_mesh_usage ; 
}
std::map<unsigned int, std::vector<unsigned int> >& GMeshLib::getMeshNodes()
{
    return m_mesh_nodes ; 
}

void GMeshLib::reportMeshUsage_(std::ostream& out) const 
{
     typedef std::map<unsigned int, unsigned int>::const_iterator MUUI ; 
     out << " meshIndex, nvert, nface, nodeCount, nodeCount*nvert, nodeCount*nface, meshName " << std::endl ; 

     unsigned tnode(0) ; 
     unsigned tvert(0) ; 
     unsigned tface(0) ; 

     for(MUUI it=m_mesh_usage.begin() ; it != m_mesh_usage.end() ; it++)
     {
         unsigned int meshIndex = it->first ; 
         unsigned int nodeCount = it->second ; 
 
         const GMesh* mesh = getMesh(meshIndex);
         const char* meshName = mesh->getName() ; 
         unsigned int nvert = mesh->getNumVertices() ; 
         unsigned int nface = mesh->getNumFaces() ; 

         //printf("  %4d (v%5d f%5d) : %6d : %7d : %7d : %s \n", meshIndex, nvert, nface, nodeCount, nodeCount*nvert, nodeCount*nface, meshName);

         out << "  " << std::setw(4) << meshIndex 
             << " ("
             << " v" << std::setw(5) << nvert 
             << " f" << std::setw(5) << nface
             << " )"
             << " : " << std::setw(7) << nodeCount
             << " : " << std::setw(10) << nodeCount*nvert
             << " : " << std::setw(10) << nodeCount*nface
             << " : " << meshName 
             << std::endl ; 


         tnode += nodeCount ; 
         tvert += nodeCount*nvert ; 
         tface += nodeCount*nface ; 
     }

     //printf(" tnode : %7d \n", tnode);
     //printf(" tvert : %7d \n", tvert);
     //printf(" tface : %7d \n", tface);

     out << " tot "
         << " node : " << std::setw(7) << tnode
         << " vert : " << std::setw(7) << tvert
         << " face : " << std::setw(7) << tface
         << std::endl ;
}

void GMeshLib::reportMeshUsage(const char* msg) const 
{
    std::ostream& out = std::cout ; 
    out << msg << std::endl ; 
    reportMeshUsage_(out); 
}

void GMeshLib::writeMeshUsage(const char* path) const 
{
    LOG(LEVEL) << " write to " << path ; 
    std::ofstream out(path); 
    out << "GMeshLib::writeMeshUsage"  << std::endl ; 
    reportMeshUsage_(out); 
}

void GMeshLib::saveMeshUsage(const char* idpath) const 
{
    std::string path = BFile::FormPath(idpath, m_reldir, "MeshUsage.txt") ; 
    writeMeshUsage(path.c_str()); 
}


