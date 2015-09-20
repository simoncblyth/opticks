#include "GGeo.hh"

#include "GCache.hh"
#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GMaterial.hh"
#include "GSolid.hh"
#include "GMesh.hh"
#include "GBoundaryLib.hh"
#include "GSensorList.hh"
#include "GSensor.hh"
#include "GMergedMesh.hh"
#include "GColors.hh"
#include "GItemIndex.hh"

#include "assert.h"
#include "stdio.h"
#include "string.h"

#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


#define BSIZ 50


GGeo* GGeo::load(const char* idpath)
{
    bool loaded = true ; 
    GGeo* ggeo = new GGeo(loaded);
    ggeo->loadMergedMeshes(idpath);
    return ggeo ; 
}



const char* GGeo::GMERGEDMESH = "GMergedMesh" ; 

void GGeo::removeMergedMeshes(const char* idpath )
{
   fs::path cachedir(idpath);

   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        fs::path mmdir(cachedir / GMERGEDMESH / boost::lexical_cast<std::string>(ridx) );
        if(fs::exists(mmdir) && fs::is_directory(mmdir))
        {   
            unsigned long nrm = fs::remove_all(mmdir);
            LOG(info) << "GGeo::removeMergedMeshes " << mmdir.string() 
                      << " removed " << nrm 
                      ; 
        }
   } 
}



void GGeo::loadMergedMeshes(const char* idpath )
{
   GCache* gc = GCache::getInstance();

   fs::path cachedir(idpath);

   for(unsigned int ridx=0 ; ridx < MAX_MERGED_MESH ; ++ridx)
   {   
        fs::path mmdir(cachedir / GMERGEDMESH / boost::lexical_cast<std::string>(ridx) );
        if(fs::exists(mmdir) && fs::is_directory(mmdir))
        {   
            const char* path = mmdir.string().c_str() ;
            LOG(debug) << "GGeo::loadMergedMeshes " << gc->getRelativePath(path) ;
            m_merged_mesh[ridx] = GMergedMesh::load( path, ridx);
        }
        else
        {
            LOG(debug) << "GGeo::loadMergedMeshes " 
                       << " no mmdir for ridx " << ridx 
                       ;
        }
   }
   LOG(info) << "GGeo::loadMergedMeshes" 
             << " loaded "  << m_merged_mesh.size()
             ;
}

void GGeo::saveMergedMeshes(const char* idpath)
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

GMergedMesh* GGeo::makeMergedMesh(unsigned int index, GNode* base)
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end())
    {
        m_merged_mesh[index] = GMergedMesh::create(index, this, base);
    }
    return m_merged_mesh[index] ;
}

unsigned int GGeo::getNumMergedMesh()
{
    return m_merged_mesh.size();
}

GMergedMesh* GGeo::getMergedMesh(unsigned int index)
{
    if(m_merged_mesh.find(index) == m_merged_mesh.end()) return NULL ;
    return m_merged_mesh[index] ;
}





void GGeo::init()
{
   if(m_loaded) return ; 

   m_boundary_lib = new GBoundaryLib();

   m_sensor_list = new GSensorList();

   // chroma/chroma/geometry.py
   // standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
   //
   GDomain<float>* standard_wavelengths = new GDomain<float>(60.f, 810.f, 20.f );  
   m_boundary_lib->setStandardDomain( standard_wavelengths );

   m_meshindex = new GItemIndex("MeshIndex") ; 
}


GGeo::~GGeo()
{
   delete m_low ; 
   delete m_high ; 
   delete m_boundary_lib ;
}

void GGeo::setLow(const gfloat3& low)
{
    m_low = new gfloat3(low);
}
void GGeo::setHigh(const gfloat3& high)
{
    m_high = new gfloat3(high);
}

void GGeo::setPath(const char* path)
{
   m_path = strdup(path);
}
void GGeo::setQuery(const char* query)
{
   m_query = strdup(query);
}
void GGeo::setCtrl(const char* ctrl)
{
   m_ctrl = strdup(ctrl);
}
void GGeo::setIdentityPath(const char* idpath)
{
   m_idpath = strdup(idpath);
}

void GGeo::updateBounds(GNode* node)
{
    if(!m_low)  m_low  = new gfloat3(1e10f, 1e10f, 1e10f) ;
    if(!m_high) m_high = new gfloat3(-1e10f, -1e10f, -1e10f) ;
  
    node->updateBounds(*m_low, *m_high);
}


void GGeo::Summary(const char* msg)
{
    printf("%s ms %lu so %lu mt %lu bs %lu ss %lu \n", msg, m_meshes.size(), m_solids.size(), m_materials.size(), m_border_surfaces.size(), m_skin_surfaces.size() );  

    if(m_low)  printf("    low  %10.3f %10.3f %10.3f \n", m_low->x, m_low->y, m_low->z);
    if(m_high) printf("    high %10.3f %10.3f %10.3f \n", m_high->x, m_high->y, m_high->z);
}

void GGeo::Details(const char* msg)
{
    printf("%s  #border_surfaces %lu #skin_surfaces %lu #materials %lu \n", msg, m_border_surfaces.size(),  m_skin_surfaces.size(), m_materials.size()); 
    char mbuf[BSIZ];

    /*
    for(unsigned int ims=0 ; ims < m_meshes.size()  ; ims++ )
    {
        GMesh* ms = m_meshes[ims];
        snprintf(mbuf,BSIZ, "%s ms %u", msg, ims);
        ms->Summary(mbuf);
    }
    */

    for(unsigned int ibs=0 ; ibs < m_border_surfaces.size()  ; ibs++ )
    {
        GBorderSurface* bs = m_border_surfaces[ibs];
        snprintf(mbuf,BSIZ, "%s bs %u", msg, ibs);
        bs->Summary(mbuf);
    }
    for(unsigned int iss=0 ; iss < m_skin_surfaces.size()  ; iss++ )
    {
        GSkinSurface* ss = m_skin_surfaces[iss];
        snprintf(mbuf,BSIZ, "%s ss %u", msg, iss);
        ss->Summary(mbuf);
    }
    for(unsigned int imat=0 ; imat < m_materials.size()  ; imat++ )
    {
        GMaterial* mat = m_materials[imat];
        snprintf(mbuf,BSIZ, "%s mt %u", msg, imat);
        mat->Summary(mbuf);
    }

    /*
    for(unsigned int isol=0 ; isol < m_solids.size()  ; isol++ )
    {
        GSolid* sol = m_solids[isol];
        snprintf(mbuf,BSIZ, "%s so %u", msg, isol);
        sol->Summary(mbuf);
    }
    */
}


void GGeo::add(GMesh* mesh)
{
    m_meshes.push_back(mesh);

    const char* name = mesh->getName();
    unsigned int index = mesh->getIndex();

    LOG(debug) << "GGeo::add (GMesh)"
              << " index " << std::setw(4) << index 
              << " name " << name 
              ;

    m_meshindex->add(name, index); 
}

void GGeo::add(GSolid* solid)
{
    m_solids.push_back(solid);
    unsigned int index = solid->getIndex(); // absolute node index, independent of the selection
    //printf("GGeo::add solid %u \n", index);
    m_solidmap[index] = solid ; 

    GSolid* check = getSolid(index);
    assert(check == solid);
}


void GGeo::dumpRaw(const char* msg)
{
    printf("%s\n", msg);     
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        mat->Summary();
    }
}


GSolid* GGeo::getSolid(unsigned int index)
{
    GSolid* solid = NULL ; 
    if(m_solidmap.find(index) != m_solidmap.end()) 
    {
        solid = m_solidmap[index] ;
        assert(solid->getIndex() == index);
    }
    return solid ; 
}



#if 0

// cannot do this check any more in GBoundary approach 

void GGeo::materialConsistencyCheck()
{
    GSolid* solid = getSolid(0);
    assert(solid);
    unsigned int nok = materialConsistencyCheck(solid);
    printf("GGeo::materialConsistencyCheck nok %u \n", nok );
}

unsigned int GGeo::materialConsistencyCheck(GSolid* solid)
{
    assert(solid);
    //solid->Summary(NULL);

    GSolid* parent = dynamic_cast<GSolid*>(solid->getParent()) ; 

    unsigned int nok = 0 ;
    if(parent)
    {
        assert(parent->getInnerMaterial() == solid->getOuterMaterial());
        nok += 1 ;
    } 
    else
    {
        assert(solid->getIndex() == 0); 
    } 

    for(unsigned int i=0 ; i < solid->getNumChildren() ; i++)
    {
        GSolid* child = dynamic_cast<GSolid*>(solid->getChild(i)) ;
        assert(child); 
        nok += materialConsistencyCheck(child);
    }
    return nok ;
}

#endif





GMaterial* GGeo::getMaterial(unsigned int aindex)
{
    GMaterial* mat = NULL ; 
    for(unsigned int i=0 ; i < m_materials.size() ; i++ )
    { 
        if(m_materials[i]->getIndex() == aindex )
        {
            mat = m_materials[i] ; 
            break ; 
        }
    }
    return mat ;
}


GPropertyMap<float>* GGeo::findRawMaterial(const char* shortname)
{
    GMaterial* mat = NULL ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++ )
    { 
        std::string sn = m_materials_raw[i]->getShortNameString();
        //printf("GGeo::findRawMaterial %d %s \n", i, sn.c_str()); 
        if(strcmp(sn.c_str(), shortname)==0)
        {
            mat = m_materials_raw[i] ; 
            break ; 
        }
    }
    return (GPropertyMap<float>*)mat ;
}


void GGeo::addToIndex(GPropertyMap<float>* psrc)
{
    unsigned int pindex = psrc->getIndex();
    if(pindex < UINT_MAX)
    {
         if(m_index.count(pindex) == 0) 
               m_index[pindex] = psrc->getShortName(); 
         else
               assert(strcmp(m_index[pindex].c_str(), psrc->getShortName()) == 0);
    }
}


void  GGeo::dumpIndex(const char* msg)
{
    printf("%s\n", msg);
    for(Index_t::iterator it=m_index.begin() ; it != m_index.end() ; it++)
         printf("  %3u :  %s \n", it->first, it->second.c_str() );
}



GProperty<float>* GGeo::findRawMaterialProperty(const char* shortname, const char* propname)
{
    GPropertyMap<float>* mat = findRawMaterial(shortname);

    GProperty<float>* prop = mat->getProperty(propname);
    prop->Summary();

    // hmm should have permanent slot in idpath 
    return prop ;   
}




GMesh* GGeo::getMesh(unsigned int aindex)
{
    GMesh* mesh = NULL ; 
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







GSkinSurface* GGeo::findSkinSurface(const char* lv)
{
    GSkinSurface* ss = NULL ; 
    for(unsigned int i=0 ; i < m_skin_surfaces.size() ; i++ )
    {
         GSkinSurface* s = m_skin_surfaces[i];
         if(s->matches(lv))   
         {
            ss = s ; 
            break ; 
         } 
    }
    return ss ;
}

GBorderSurface* GGeo::findBorderSurface(const char* pv1, const char* pv2)
{
    GBorderSurface* bs = NULL ; 
    for(unsigned int i=0 ; i < m_border_surfaces.size() ; i++ )
    {
         GBorderSurface* s = m_border_surfaces[i];
         if(s->matches(pv1,pv2))   
         {
            bs = s ; 
            break ; 
         } 
    }
    return bs ;
}



void GGeo::dumpRawSkinSurface(const char* name)
{
    GSkinSurface* ss = NULL ; 
    unsigned int n = getNumRawSkinSurfaces();
    for(unsigned int i = 0 ; i < n ; i++)
    {
        ss = m_skin_surfaces_raw[i];
        ss->Summary("GGeo::dumpRawSkinSurface", 10); 
    }
}

void GGeo::dumpRawBorderSurface(const char* name)
{
    GBorderSurface* bs = NULL ; 
    unsigned int n = getNumRawBorderSurfaces();
    for(unsigned int i = 0 ; i < n ; i++)
    {
        bs = m_border_surfaces_raw[i];
        bs->Summary("GGeo::dumpRawBorderSurface", 10); 
    }
}


void GGeo::sensitize(const char* idpath, const char* ext)
{
    m_sensor_list->load(idpath, ext);

    LOG(info) << "GGeo::sensitize " << m_sensor_list->description() ; 

    GSolid* root = getSolid(0);

    sensitize_traverse(root, 0 );

    LOG(info) << "GGeo::sensitize sensitize_count " << m_sensitize_count  ; 
}

void GGeo::sensitize_traverse( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    unsigned int nodeIndex = node->getIndex();

    GSensor* sensor = m_sensor_list->findSensorForNode(nodeIndex);

    if(sensor)
    {
        m_sensitize_count++ ; 

        solid->setSensor(sensor);  
        //LOG(info) << "[" << std::setw(5) << m_sensitize_count << "] " << sensor->description() ; 
    }
    else
    {
        // every triangle needs an unsigned int, for non-sensitized provide a 0 (which means real indices must be 1-based)
        solid->setSensor(NULL);  
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) sensitize_traverse(node->getChild(i), depth + 1);
}







void GGeo::dumpRawMaterialProperties(const char* msg)
{
    printf("%s\n", msg);     
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        //mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }
}


void GGeo::findScintillators(const char* props)
{
    m_scintillators_raw = getRawMaterialsWithProperties(props, ",");
    assert(m_scintillators_raw.size() > 0 );
}
void GGeo::dumpScintillators(const char* msg)
{
    LOG(info)<< msg ;
    for(unsigned int i=0; i<m_scintillators_raw.size() ; i++)
    {
        GMaterial* mat = m_scintillators_raw[i];
        //mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }              
}

unsigned int GGeo::getNumScintillators()
{
    return m_scintillators_raw.size();
}

GMaterial* GGeo::getScintillator(unsigned int index)
{
    return index < m_scintillators_raw.size() ? m_scintillators_raw[index] : NULL ; 
}


std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, const char* delim)
{
    std::vector<std::string> elem ;
    boost::split(elem, props, boost::is_any_of(delim));

    std::vector<GMaterial*>  selected ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        unsigned int found(0);
        for(unsigned int p=0 ; p < elem.size() ; p++)
        { 
           if(mat->hasProperty(elem[p].c_str())) found+=1 ;        
        }
        if(found == elem.size()) selected.push_back(mat);
    }
    return selected ;  
}





void GGeo::countMeshUsage(unsigned int meshIndex, unsigned int nodeIndex, const char* lv, const char* pv)
{
     // called during GGeo creation from: void AssimpGGeo::convertStructure(GGeo* gg)
     //printf("GGeo::countMeshUsage %d %d %s %s \n", meshIndex, nodeIndex, lv, pv);
     m_mesh_usage[meshIndex] += 1 ; 
}

void GGeo::reportMeshUsage(const char* msg)
{
     printf("%s\n", msg);
     unsigned int tv(0) ; 
     typedef std::map<unsigned int, unsigned int>::const_iterator MUUI ; 
     for(MUUI it=m_mesh_usage.begin() ; it != m_mesh_usage.end() ; it++)
     {
         unsigned int meshIndex = it->first ; 
         unsigned int nodeCount = it->second ; 
 
         GMesh* mesh = getMesh(meshIndex);
         const char* meshName = mesh->getName() ; 
         unsigned int nv = mesh->getNumVertices() ; 
         unsigned int nf = mesh->getNumFaces() ; 

         printf("  %4d (v%5d f%5d) : %6d : %7d : %s \n", meshIndex, nv, nf, nodeCount, nodeCount*nv, meshName);
         tv += nodeCount*nv ; 
     }
     printf(" tv : %7d \n", tv);
}


void GGeo::dumpStats(const char* msg)
{
    printf("%s\n", msg);

    unsigned int nmm = getNumMergedMesh();

    unsigned int totVertices(0);
    unsigned int totFaces(0);
    unsigned int vtotVertices(0);
    unsigned int vtotFaces(0);

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = getMergedMesh(i);
        GBuffer* tbuf = mm->getTransformsBuffer();
        GBuffer* vbuf = mm->getVerticesBuffer();
        GBuffer* ibuf = mm->getIndicesBuffer();

        unsigned int numVertices = vbuf->getNumItems() ;
        unsigned int numFaces = ibuf->getNumItems()/3;
        unsigned int numTransforms = tbuf ? tbuf->getNumItems() : 1  ;

        printf(" mm %2d : vertices %7d faces %7d transforms %7d \n", i, numVertices, numFaces, numTransforms);

        totVertices += numVertices ; 
        totFaces    += numFaces ; 

        vtotVertices += numVertices*numTransforms ; 
        vtotFaces    += numFaces*numTransforms ; 
    } 

    printf("   totVertices %9d  totFaces %9d \n", totVertices, totFaces );
    printf("  vtotVertices %9d vtotFaces %9d (virtual: scaling by transforms)\n", vtotVertices, vtotFaces );
    printf("  vfacVertices %9.3f vfacFaces %9.3f (virtual to total ratio)\n", float(vtotVertices)/float(totVertices), float(vtotFaces)/float(totFaces) );



}

gfloat4 GGeo::getCenterExtent(unsigned int target, unsigned int merged_mesh_index )
{
    GMergedMesh* mm = getMergedMesh(merged_mesh_index);
    assert(mm);
    return mm->getCenterExtent(target); 
}

