#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "BStr.hh"
#include "BFile.hh"

#include "NLODConfig.hpp"

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


const char* GGeoLib::RelDir(const char* name, bool analytic) // static
{
    return analytic ? BStr::concat(name, "Analytic", NULL) : name ; 
}

const char* GGeoLib::getRelDir(const char* name) const 
{
    return RelDir(name, m_analytic);
}


GBndLib* GGeoLib::getBndLib() const 
{
    return m_bndlib ; 
}


GGeoLib* GGeoLib::Load(Opticks* opticks, bool analytic, GBndLib* bndlib)
{
    GGeoLib* glib = new GGeoLib(opticks, analytic, bndlib);
    glib->loadFromCache();
    return glib ; 
}

GGeoLib::GGeoLib(Opticks* ok, bool analytic, GBndLib* bndlib) 
    :
    m_ok(ok),
    m_lodconfig(m_ok->getLODConfig()), 
    m_lod(m_ok->getLOD()), 
    m_analytic(analytic),
    m_bndlib(bndlib),
    m_mesh_version(NULL),
    m_verbosity(m_ok->getVerbosity())
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



bool GGeoLib::HasCacheConstituent(const char* idpath, bool analytic, unsigned ridx) 
{
    const char* sidx = BStr::itoa(ridx) ;
    // reldir depends on m_analytic
    std::string smmpath = BFile::FormPath(idpath, RelDir(GMERGEDMESH, analytic), sidx );
    std::string sptpath = BFile::FormPath(idpath, RelDir(GPARTS, analytic),      sidx );

    const char* mmpath = smmpath.c_str();
    const char* ptpath = sptpath.c_str();

    return BFile::ExistsDir(mmpath) && BFile::ExistsDir(ptpath) ; 
}


void GGeoLib::removeConstituents(const char* idpath )
{
   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   

        const char* sidx = BStr::itoa(ridx) ;
        // reldir depends on m_analytic
        std::string smmpath = BFile::FormPath(idpath, getRelDir(GMERGEDMESH), sidx );
        std::string sptpath = BFile::FormPath(idpath, getRelDir(GPARTS),      sidx );

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
* GMergedMesh geocode ANA/TRI is set according to the m_analytic toggle
* m_analytic toggle also changes the directory that gets loaded 

  * this is kinda funny because GParts is always analytic 

**/

void GGeoLib::loadConstituents(const char* idpath )
{
   LOG(LEVEL) 
       << " mm.reldir " << getRelDir(GMERGEDMESH)
       << " gp.reldir " << getRelDir(GPARTS)
       << " MAX_MERGED_MESH  " << MAX_MERGED_MESH 
       ; 

   LOG(LEVEL) << idpath ;

   std::stringstream ss ; 

   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        const char* sidx = BStr::itoa(ridx) ;
        std::string smmpath = BFile::FormPath(idpath, getRelDir(GMERGEDMESH), sidx );
        std::string sptpath = BFile::FormPath(idpath, getRelDir(GPARTS),      sidx );
        std::string sptspath = BFile::FormPath(idpath, getRelDir(GPTS),       sidx );

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
            
            mm->setGeoCode( m_analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED );  // assuming uniform : all analytic/triangulated GPU geom

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
        mm->save(idpath, getRelDir(GMERGEDMESH), sidx ); 


        GParts* pt = mm->getParts() ; 
        if(pt)  
        {          
           std::string partsp_ = BFile::FormPath(idpath, getRelDir(GPARTS), sidx );
           const char* partsp = partsp_.c_str();
           pt->save(partsp); 
        }


        GPts* pts = mm->getPts() ; 
        if(pts)  
        {          
           std::string ptsp_ = BFile::FormPath(idpath, getRelDir(GPTS), sidx );
           const char* ptsp = ptsp_.c_str();
           pts->save(ptsp); 
        }

    }
}



GMergedMesh* GGeoLib::makeMergedMesh(unsigned index, GNode* base, GNode* root, unsigned verbosity )
{
    LOG(LEVEL) << " mm " << index ;  

    if(m_merged_mesh.find(index) == m_merged_mesh.end())
    {
        m_merged_mesh[index] = GMergedMesh::Create(index, base, root, verbosity );
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
       << ( m_analytic ? " ANALYTIC " : " TRIANGULATED " ) 
       << " numMergedMesh " << getNumMergedMesh()
       << " ptr " << this
       ; 

    return ss.str();
}


void GGeoLib::dump(const char* msg)
{
    LOG(info) << msg << " " << desc() ; 

    unsigned nmm = getNumMergedMesh();
    unsigned num_total_volumes = 0 ; 
    unsigned num_instanced_volumes = 0 ; 

    for(unsigned i=0 ; i < nmm ; i++)
    {   
        GMergedMesh* mm = getMergedMesh(i); 

        unsigned numVolumes = mm ? mm->getNumVolumes() : -1 ;
        unsigned numITransforms = mm ? mm->getNumITransforms() : -1 ;
        if( i == 0 ) num_total_volumes = numVolumes ; 
        std::cout << GMergedMesh::Desc(mm) << std::endl ; 
        num_instanced_volumes += i > 0 ? numITransforms*numVolumes : 0 ;
    }
    std::cout
                << " num_total_volumes " << num_total_volumes 
                << " num_instanced_volumes " << num_instanced_volumes 
                << " num_global_volumes " << num_total_volumes - num_instanced_volumes
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





