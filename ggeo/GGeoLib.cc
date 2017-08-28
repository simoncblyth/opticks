#include <cstring>

#include "BStr.hh"
#include "BFile.hh"


#include "Opticks.hh"

#include "GGeoLib.hh"
#include "GMergedMesh.hh"
#include "GParts.hh"
#include "GNode.hh"


#include "GGEO_BODY.hh"
#include "PLOG.hh"
// trace/debug/info/warning/error/fatal


const char* GGeoLib::GMERGEDMESH = "GMergedMesh" ; 
const char* GGeoLib::GPARTS = "GParts" ; 

const char* GGeoLib::getRelDir(const char* name)
{
    return m_analytic ? BStr::concat(name, "Analytic", NULL) : name ; 
}

GGeoLib* GGeoLib::Load(Opticks* opticks, bool analytic)
{
    GGeoLib* glib = new GGeoLib(opticks, analytic);
    glib->loadFromCache();
    return glib ; 
}


std::string GGeoLib::desc() const 
{
    std::stringstream ss ; 

    ss << "GGeoLib"
       << ( m_analytic ? " ANALYTIC " : " TRIANGULATED " ) 
       << " numMergedMesh " << getNumMergedMesh()
       ; 

    return ss.str();
}


GGeoLib::GGeoLib(Opticks* opticks, bool analytic) 
    :
    m_opticks(opticks),
    m_analytic(analytic),
    m_mesh_version(NULL)
{
}

unsigned int GGeoLib::getNumMergedMesh() const 
{
    return m_merged_mesh.size();
}
void GGeoLib::setMeshVersion(const char* mesh_version)
{
    m_mesh_version = mesh_version ? strdup(mesh_version) : NULL ;
}
const char* GGeoLib::getMeshVersion() const 
{
    return m_mesh_version ;
}


GMergedMesh* GGeoLib::getMergedMesh(unsigned int index)
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end()) return NULL ;
    GMergedMesh* mm = m_merged_mesh[index] ;
    return mm ; 
}

void GGeoLib::loadFromCache()
{
    const char* idpath = m_opticks->getIdPath() ;
    LOG(debug) << "GGeoLib::loadFromCache" ;
    loadConstituents(idpath);
}

void GGeoLib::saveToCache()
{
    const char* idpath = m_opticks->getIdPath() ;
    saveConstituents(idpath);
}

void GGeoLib::removeConstituents(const char* idpath )
{
   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        const char* sidx = BStr::itoa(ridx) ;
        std::string smmpath = BFile::FormPath(idpath, getRelDir(GMERGEDMESH), sidx );
        std::string sptpath = BFile::FormPath(idpath, getRelDir(GPARTS),      sidx );

        const char* mmpath = smmpath.c_str();
        const char* ptpath = sptpath.c_str();

        if(BFile::ExistsDir(mmpath))
        {   
            BFile::RemoveDir(mmpath); 
            LOG(info) << "GGeoLib::removeConstituents " << mmpath ; 
        }

        if(BFile::ExistsDir(ptpath))
        {   
            BFile::RemoveDir(ptpath); 
            LOG(info) << "GGeoLib::removeConstituents " << mmpath ; 
        }
   } 
}

void GGeoLib::loadConstituents(const char* idpath )
{
   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        const char* sidx = BStr::itoa(ridx) ;
        std::string smmpath = BFile::FormPath(idpath, getRelDir(GMERGEDMESH), sidx );
        std::string sptpath = BFile::FormPath(idpath, getRelDir(GPARTS),      sidx );

        const char* mmpath = smmpath.c_str();
        const char* ptpath = sptpath.c_str();

        GMergedMesh* mm = BFile::ExistsDir(mmpath) ? GMergedMesh::load( mmpath, ridx, m_mesh_version ) : NULL ; 
        GParts*      pt = BFile::ExistsDir(ptpath) ? GParts::Load( ptpath ) : NULL ; 

        if( mm )
        {
            mm->setParts(pt);
            m_merged_mesh[ridx] = mm ; 
        }
        else
        {
            if(pt) LOG(fatal) << "GGeoLib::loadConstituents"
                              << " pt exists but mm doesn not " ;
            assert(pt==NULL);
            LOG(debug) << "GGeoLib::loadConstituents " 
                       << " no mmdir for ridx " << ridx 
                       ;
        }
   }
   LOG(debug) << "GGeoLib::loadConstituents" 
             << " loaded "  << m_merged_mesh.size()
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
        mm->save(idpath, getRelDir(GMERGEDMESH), sidx ); 

        std::string sptpath = BFile::FormPath(idpath, getRelDir(GPARTS), sidx );
        const char* ptpath = sptpath.c_str();

        GParts* pt = mm->getParts() ; 
        if(pt)  pt->save(ptpath); 

    }
}

GMergedMesh* GGeoLib::makeMergedMesh(unsigned int index, GNode* base, GNode* root, unsigned verbosity )
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end())
    {
        m_merged_mesh[index] = GMergedMesh::create(index, base, root, verbosity );
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


