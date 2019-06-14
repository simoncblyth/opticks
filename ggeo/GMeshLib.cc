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
#ifdef OLD_INDEX
#include "GItemIndex.hh"   // <-- aim to remove 
#endif
#include "GItemList.hh"

#include "PLOG.hh"

const plog::Severity GMeshLib::LEVEL = debug ; 


const unsigned GMeshLib::MAX_MESH = 250 ;   // <-- hmm 500 too large ? it means a lot of filesystem checking 


#ifdef OLD_INDEX
const char* GMeshLib::GITEMINDEX = "GItemIndex" ; 
const char* GMeshLib::GMESHLIB_INDEX = "MeshIndex" ; 
#endif

const char* GMeshLib::GMESHLIB_LIST = "GMeshLib" ; 


const char* GMeshLib::GMESHLIB = "GMeshLib" ; 
const char* GMeshLib::GMESHLIB_NCSG = "GMeshLibNCSG" ; 


GMeshLib::GMeshLib(Opticks* ok ) 
   :
   m_ok(ok),
   m_direct(ok->isDirect()),
   m_reldir(GMESHLIB),
   m_reldir_solids(GMESHLIB_NCSG),
#ifdef OLD_INDEX
   m_meshindex(new GItemIndex(GITEMINDEX, GMESHLIB_INDEX )),
#endif
   m_meshnames(new GItemList("GMeshLib", "GItemList")),
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

#ifdef OLD_INDEX
    m_meshindex = GItemIndex::load(idpath, GITEMINDEX, GMESHLIB_INDEX ) ;
    assert(m_meshindex);
    bool has_index = m_meshindex->hasIndex() ;
    if(!has_index)  LOG(fatal) << " meshindex load failure " ; 
    assert(has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G ");
#endif


    m_meshnames = GItemList::Load(idpath, GMESHLIB_LIST, "GItemList" ) ;
    assert(m_meshnames);

    loadMeshes(idpath);
}

void GMeshLib::save() const
{
    const char* idpath = m_ok->getIdPath() ;


#ifdef OLD_INDEX
    assert( m_meshindex ); 
    m_meshindex->save(idpath);
#endif

    assert( m_meshnames ); 
    m_meshnames->save(idpath );

    saveMeshes(idpath);
    saveMeshUsage(idpath);
}


#ifdef OLD_INDEX
GItemIndex* GMeshLib::getMeshIndex()
{
    return m_meshindex ; 
}
#endif

const char* GMeshLib::getMeshName(unsigned aindex) const 
{
#ifdef OLD_INDEX
    return m_meshindex->getNameSource(aindex);
#else
    return m_meshnames->getKey(aindex); 
#endif
}

#ifdef OLD_INDEX
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
        assert(name);
        unsigned aindex = m_meshindex->getIndexSource(name, m_missing); // huh : why have to go via name : consistency check probably 
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
#else
void GMeshLib::dump(const char* msg) const
{
    unsigned num_mesh = m_meshnames->getNumKeys();
    LOG(info) << msg 
              << " meshnames " << num_mesh 
              << " meshes " << m_meshes.size() 
              ; 

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        const char* name = getMeshName(i); 
        assert(name);
        bool startswith(false); 
        int aidx = getMeshIndexWithName(name, startswith);  // hmm this is lib index not mesh index

        const GMesh* mesh = getMeshSimple(i);
        assert( mesh ); 
        unsigned midx = mesh->getIndex();  

        assert( strcmp(mesh->getName(), name) == 0);

        std::cout 
               << " i " << std::setw(3) << i  
               << " aidx " << std::setw(3) << aidx  
               << " midx " << std::setw(3) << midx  
               << " name " << std::setw(50) << name 
               << " mesh " << mesh->desc()
               << std::endl 
               ;
    }
}
#endif


unsigned GMeshLib::getNumMeshes() const 
{
    return m_meshes.size();
}

unsigned GMeshLib::getNumSolids() const 
{
    return m_solids.size();
}



const NCSG* GMeshLib::getSolidWithIndex(unsigned aindex) const  // gets the solid with the index
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

const GMesh* GMeshLib::getMeshWithIndex(unsigned aindex) const // gets the mesh with the index
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


const GMesh* GMeshLib::getMeshWithName(const char* name, bool startswith) const 
{
    int idx = getMeshIndexWithName(name, startswith);
    return idx == -1 ? NULL : getMeshWithIndex(idx);
}


int GMeshLib::getMeshIndexWithName(const char* name, bool startswith)  const 
{
#ifdef OLD_INDEX
    unsigned aindex = startswith ? 
          m_meshindex->getIndexSourceStarting(name, m_missing)
          :
          m_meshindex->getIndexSource(name, m_missing)
          ;
    assert( aindex != m_missing );
#else
    int aindex = startswith ? m_meshnames->findIndexWithKeyStarting(name) : m_meshnames->findIndex(name) ;   
#endif
    return aindex ; 
}


GMesh* GMeshLib::getMeshSimple(unsigned index) const 
{
    assert( index < m_meshes.size() ); 
    const GMesh* mesh_ = m_meshes[index] ; 
    GMesh* mesh = const_cast<GMesh*>(mesh_);   // hmm rethink needed ?

    bool index_match = mesh->getIndex() == index ; 
    if(!index_match)
       LOG(error) 
           << " mesh indices do not match " 
           << " m_meshes index " << index
           << " mesh.index " << mesh->getIndex()
           ; 

    //assert( index_match ); 
    return mesh ; 
}  

const GMesh* GMeshLib::getAltMesh(unsigned index) const 
{
    const GMesh* mesh = getMeshSimple(index); 
    return mesh->getAlt();  
}


void GMeshLib::getMeshIndicesWithAlt(std::vector<unsigned>& indices) const 
{
    for(unsigned i=0 ; i < m_meshes.size() ; i++ )
    {
        const GMesh* mesh = m_meshes[i] ; 
        const GMesh* alt = mesh->getAlt(); 
        if(alt) indices.push_back(i); 
    }
}


/**
GMeshLib::add
----------------

Invoked via GGeo::add from X4PhysicalVolume::convertSolids_r as each distinct 
solid is encountered in the recursive traverse.

**/

void GMeshLib::add(const GMesh* mesh)
{
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

    if(m_direct)
    { 
        assert(solid) ;                
    }
    m_solids.push_back(solid); 

#ifdef OLD_INDEX
    m_meshindex->add(name, index); 
#endif

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
            if( m_direct )
            {
                assert( soliddir_exists && "GMeshLib persisted GMesh are expected to have paired GMeshLibNCSG dirs"); 
            }

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
            if(name == NULL)
                LOG(fatal) 
                    << " no name for mesh idx " << idx 
                    ; 

            assert(name);

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
 
         const GMesh* mesh = getMeshWithIndex(meshIndex);
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


