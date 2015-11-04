#include "GGeoLib.hh"

#include "GGeo.hh"
#include "GCache.hh"
#include "GMergedMesh.hh"
#include "GNode.hh"

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

const char* GGeoLib::GMERGEDMESH = "GMergedMesh" ; 


GGeoLib* GGeoLib::load(GCache* cache)
{
    GGeoLib* glib = new GGeoLib(cache);
    glib->loadFromCache();
    return glib ; 
}


void GGeoLib::loadFromCache()
{
    const char* idpath = m_cache->getIdPath() ;
    loadMergedMeshes(idpath);
}

void GGeoLib::saveToCache()
{
    const char* idpath = m_cache->getIdPath() ;
    saveMergedMeshes(idpath);
}


void GGeoLib::removeMergedMeshes(const char* idpath )
{
   fs::path cachedir(idpath);

   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        fs::path mmdir(cachedir / GMERGEDMESH / boost::lexical_cast<std::string>(ridx) );
        if(fs::exists(mmdir) && fs::is_directory(mmdir))
        {   
            unsigned long nrm = fs::remove_all(mmdir);
            LOG(info) << "GGeoLib::removeMergedMeshes " << mmdir.string() 
                      << " removed " << nrm 
                      ; 
        }
   } 
}

void GGeoLib::loadMergedMeshes(const char* idpath )
{
   fs::path cachedir(idpath);

   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        fs::path mmdir(cachedir / GMERGEDMESH / boost::lexical_cast<std::string>(ridx) );
        if(fs::exists(mmdir) && fs::is_directory(mmdir))
        {   
            const char* path = mmdir.string().c_str() ;
            LOG(debug) << "GGeoLib::loadMergedMeshes " << m_cache->getRelativePath(path) ;
            m_merged_mesh[ridx] = GMergedMesh::load( path, ridx, m_mesh_version );
        }
        else
        {
            LOG(debug) << "GGeoLib::loadMergedMeshes " 
                       << " no mmdir for ridx " << ridx 
                       ;
        }
   }
   LOG(info) << "GGeoLib::loadMergedMeshes" 
             << " loaded "  << m_merged_mesh.size()
             ;
}

void GGeoLib::saveMergedMeshes(const char* idpath)
{
    removeMergedMeshes(idpath); // clean old meshes to avoid duplication when repeat counts go down 

    typedef std::map<unsigned int,GMergedMesh*>::const_iterator MUMI ; 
    for(MUMI it=m_merged_mesh.begin() ; it != m_merged_mesh.end() ; it++)
    {
        unsigned int ridx = it->first ; 
        GMergedMesh* mergedmesh = it->second ; 
        assert(mergedmesh->getIndex() == ridx);
        mergedmesh->save(idpath, GMERGEDMESH, boost::lexical_cast<std::string>(ridx).c_str()); 
    }
}

GMergedMesh* GGeoLib::makeMergedMesh(GGeo* ggeo, unsigned int index, GNode* base)
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end())
    {
        m_merged_mesh[index] = GMergedMesh::create(index, ggeo, base);
    }
    return m_merged_mesh[index] ;
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



