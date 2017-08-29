#include <sstream>
#include <iostream>
#include <iomanip>
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
   LOG(info) 
             << "GGeoLib::loadConstituents"
             << " idpath " << idpath 
             ;
             
   LOG(info) 
             << "GGeoLib::loadConstituents"
             << " mm.reldir " << getRelDir(GMERGEDMESH)
             << " gp.reldir " << getRelDir(GPARTS)
             << " MAX_MERGED_MESH  " << MAX_MERGED_MESH 
             ; 


   std::stringstream ss ; 

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
            ss << std::setw(3) << ridx << "," ; 
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
   LOG(info) << "GGeoLib::loadConstituents" 
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




std::string GGeoLib::desc() const 
{
    std::stringstream ss ; 

    ss << "GGeoLib"
       << ( m_analytic ? " ANALYTIC " : " TRIANGULATED " ) 
       << " numMergedMesh " << getNumMergedMesh()
       ; 

    return ss.str();
}


void GGeoLib::dump(const char* msg)
{
    LOG(info) << msg ; 
    LOG(info) << desc() ; 

    unsigned nmm = getNumMergedMesh();

    for(unsigned i=0 ; i < nmm ; i++)
    {   
        GMergedMesh* mm = getMergedMesh(i); 
        const char geocode = mm ? mm->getGeoCode() : '-' ;
        unsigned numSolids = mm ? mm->getNumSolids() : -1 ;
        unsigned numFaces = mm ? mm->getNumFaces() : -1 ;
        unsigned numITransforms = mm ? mm->getNumITransforms() : -1 ;

        std::cout << "mm" 
                  << " i " << std::setw(3) << i 
                  << " geocode " << std::setw(3) << geocode 
                  << std::setw(5) << ( mm ? " " : "NULL" ) 
                  << std::setw(5) << ( mm && mm->isSkip(  ) ? "SKIP" : " " ) 
                  << std::setw(7) << ( mm && mm->isEmpty()  ? "EMPTY" : " " ) 
                  << " numSolids " << std::setw(10) << numSolids
                  << " numFaces  " << std::setw(10) << numFaces
                  << " numITransforms  " << std::setw(10) << numITransforms
                  << std::endl
                  ;   

    }
}


