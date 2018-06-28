#include <ostream>
#include <fstream>
#include <iomanip>

#include "BStr.hh"
#include "BFile.hh"
#include "Opticks.hh"

#include "GMesh.hh"
#include "GMeshLib.hh"
#include "GItemIndex.hh"

#include "PLOG.hh"


const unsigned GMeshLib::MAX_MESH = 500 ; 
const char* GMeshLib::GITEMINDEX = "GItemIndex" ; 
const char* GMeshLib::GMESHLIB_INDEX = "MeshIndex" ; 
const char* GMeshLib::GMESHLIB_INDEX_ANALYTIC = "MeshIndexAnalytic" ; 

const char* GMeshLib::GMESHLIB = "GMeshLib" ; 
const char* GMeshLib::GMESHLIB_ANALYTIC = "GMeshLibAnalytic" ; 

const char* GMeshLib::GetRelDir(bool analytic)
{
    return analytic ? GMESHLIB_ANALYTIC : GMESHLIB ;
}
const char* GMeshLib::GetRelDirIndex(bool analytic)
{
    return analytic ? GMESHLIB_INDEX_ANALYTIC : GMESHLIB_INDEX  ;
}


GMeshLib::GMeshLib(Opticks* ok, bool analytic) 
   :
   m_ok(ok),
   m_analytic(analytic),
   m_reldir(strdup(GetRelDir(analytic))),
   m_meshindex(NULL),
   m_missing(std::numeric_limits<unsigned>::max())
{
}


bool GMeshLib::isAnalytic() const 
{
    return m_analytic ; 
}

GMeshLib* GMeshLib::Load(Opticks* ok, bool analytic)
{
    GMeshLib* meshlib = new GMeshLib(ok, analytic);
    meshlib->loadFromCache(); 
    return meshlib ; 
}

void GMeshLib::loadFromCache()
{
    const char* idpath = m_ok->getIdPath() ;
    m_meshindex = GItemIndex::load(idpath, GITEMINDEX, GetRelDirIndex(m_analytic)) ;
    assert(m_meshindex);

    bool has_index = m_meshindex->hasIndex() ;
    if(!has_index)  LOG(fatal) << " meshindex load failure " ; 
    //assert(has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G ");

    loadMeshes(idpath);
}

void GMeshLib::save() const
{
    const char* idpath = m_ok->getIdPath() ;

    if(m_meshindex)
    {
        m_meshindex->save(idpath);
    }
    else
    {
        LOG(warning) << "GMeshLib::save m_meshindex NULL " ; 
    }

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
               << " name " << std::setw(40) << name 
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










unsigned int GMeshLib::getNumMeshes() const 
{
    return m_meshes.size();
}

const GMesh* GMeshLib::getMesh(unsigned aindex) const 
{
    const GMesh* mesh = NULL ; 
    for(unsigned int i=0 ; i < m_meshes.size() ; i++ )
    { 
        if(m_meshes[i]->getIndex() == aindex )
        {
            mesh = m_meshes[i] ; 
            break ; 
        }
    }
    return mesh ;
}  


const GMesh* GMeshLib::getMesh(const char* name, bool startswith) const 
{
    unsigned aindex = getMeshIndex(name, startswith);
    return getMesh(aindex);
}



void GMeshLib::add(const GMesh* mesh)
{
    if(!m_meshindex) m_meshindex = new GItemIndex(GITEMINDEX, GetRelDirIndex(m_analytic))   ;

    m_meshes.push_back(mesh);

    const char* name = mesh->getName();
    unsigned int index = mesh->getIndex();

    assert(name) ; 

    LOG(debug) << "GMeshLib::add (GMesh)"
              << " index " << std::setw(4) << index 
              << " name " << name 
              ;

    m_meshindex->add(name, index); 
}




void GMeshLib::removeMeshes(const char* idpath ) const 
{
   for(unsigned int idx=0 ; idx < MAX_MESH ; ++idx)
   {   
        const char* sidx = BStr::itoa(idx);
        if(BFile::ExistsDir(idpath, m_reldir, sidx))
        { 
            BFile::RemoveDir(idpath, m_reldir, sidx); 
        }
   } 
}




void GMeshLib::loadMeshes(const char* idpath )
{
   LOG(info) << "idpath "  << idpath ;  

   // TODO: read the directory instead of just checking existance of MAX_MESH paths ?
   //       (or use the index ?)

   for(unsigned int idx=0 ; idx < MAX_MESH ; ++idx)
   {   
        const char* sidx = BStr::itoa(idx);
        std::string spath = BFile::FormPath(idpath, m_reldir, sidx);
        const char* path = spath.c_str() ;

        if(BFile::ExistsDir(path))
        {   
            LOG(debug) << "GMeshLib::loadMeshes " << path ; 
            GMesh* mesh = GMesh::load( path );

            const char* name = getMeshName(idx);
            assert(name);

            // mesh->updateBounds(); // CAN THIS BE DONE IN LOAD ? Perhaps MM complications
           // should have been done by GMesh::loadBuffers ?

            mesh->setIndex(idx);
            mesh->setName(strdup(name));

            m_meshes.push_back(mesh);
        }
        else
        {
            LOG(debug) << "GMeshLib::loadMeshes " 
                       << " no mdir for idx " << idx 
                       << " path " << path 
                       ;
        }
   }
   LOG(debug) << "GMeshLib::loadMeshes" 
             << " loaded "  << m_meshes.size()
             ;
}

void GMeshLib::saveMeshes(const char* idpath) const 
{
    removeMeshes(idpath); // clean old meshes to avoid duplication when repeat counts go down 

    typedef std::vector<const GMesh*>::const_iterator VMI ; 
    for(VMI it=m_meshes.begin() ; it != m_meshes.end() ; it++)
    {
        const GMesh* mesh = *it ; 
        unsigned int idx = mesh->getIndex() ; 
        const char* sidx = BStr::itoa(idx);

        mesh->save(idpath, m_reldir, sidx); 
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
             << " : " << std::setw(6) << nodeCount
             << " : " << std::setw(7) << nodeCount*nvert
             << " : " << std::setw(7) << nodeCount*nface
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
    LOG(info) << " write to " << path ; 
    std::ofstream out(path); 
    out << "GMeshLib::writeMeshUsage"  << std::endl ; 
    reportMeshUsage_(out); 
}

void GMeshLib::saveMeshUsage(const char* idpath) const 
{
    std::string path = BFile::FormPath(idpath, m_reldir, "MeshUsage.txt") ; 
    writeMeshUsage(path.c_str()); 
}


