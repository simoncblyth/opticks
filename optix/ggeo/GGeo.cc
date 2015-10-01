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
#include "GItemList.hh"

// npy-
#include "TorchStepNPY.hpp"

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
            m_merged_mesh[ridx] = GMergedMesh::load( path, ridx, m_mesh_version );
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

   if(m_volnames)
   {
       m_pvlist = new GItemList("PVNames") ; 
       m_lvlist = new GItemList("LVNames") ; 
   }
}

void GGeo::loadFromCache(const char* idpath)
{   
    loadMergedMeshes(idpath);
        
    m_meshindex = GItemIndex::load(idpath, "MeshIndex");

    if(m_volnames)
    {
        m_pvlist = GItemList::load(idpath, "PVNames");
        m_lvlist = GItemList::load(idpath, "LVNames");
    }
}


void GGeo::save(const char* idpath)
{
    saveMergedMeshes(idpath );

    m_meshindex->save(idpath);

    if(m_volnames)
    {
        m_pvlist->save(idpath);
        m_lvlist->save(idpath);
    }
}



GGeo* GGeo::load(const char* idpath, const char* mesh_version)
{
    bool loaded = true ; 
    bool volnames = true ; 
    GGeo* ggeo = new GGeo(loaded, volnames);
    ggeo->setMeshVersion(mesh_version);
    ggeo->loadFromCache(idpath);
    return ggeo ; 
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

    if(m_volnames)
    { 
        m_lvlist->add(solid->getLVName()); 
        m_pvlist->add(solid->getPVName()); 
    }

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

glm::vec4 GGeo::getCenterExtent(unsigned int target, unsigned int merged_mesh_index )
{
    GMergedMesh* mm = getMergedMesh(merged_mesh_index);
    assert(mm);

    glm::vec4 ce ; 
    if(merged_mesh_index == 0)
    {
        gfloat4 vce = mm->getCenterExtent(target); 
        ce.x = vce.x ; 
        ce.y = vce.y ; 
        ce.z = vce.z ; 
        ce.w = vce.w ; 
    }
    else
    {
        float* transform = mm->getTransform(target);
        ce.x = *(transform + 4*3 + 0) ; 
        ce.y = *(transform + 4*3 + 1) ; 
        ce.z = *(transform + 4*3 + 2) ; 

        gfloat4 vce = mm->getCenterExtent(0); 
        ce.w = vce.w ;  
        // somewhat dodgy, should probably find the largest extent 
        // of all the local coordinate extents
    }
    return ce ; 
}



void GGeo::targetTorchStep( TorchStepNPY* torchstep )
{
    // targetted positioning and directioning of the torch requires geometry info, 
    // which is not available within npy- so need to externally setPosition and setDirection
    // based on integer addresses specifying:
    //   
    //          (volume   index, merged mesh index=0)
    //          (instance index, merged mesh index>0)


    glm::ivec4& ipos_target = torchstep->getPosTarget() ;    
    glm::ivec4& idir_target = torchstep->getDirTarget() ;    

    if(ipos_target.x > 0 || ipos_target.y > 0) 
    {    
            glm::vec3 pos_target = glm::vec3(getCenterExtent(ipos_target.x,ipos_target.y));   
            torchstep->setPosition(pos_target);  
    }    

    if(idir_target.x > 0 || idir_target.y > 0) 
    {    
            glm::vec3 tgt = glm::vec3(getCenterExtent(idir_target.x,idir_target.y));   
            glm::vec3 pos = torchstep->getPosition();
            glm::vec3 dir = glm::normalize( tgt - pos );
            torchstep->setDirection(dir);
    }    

    glm::vec3 pol( 0.f, 0.f, 1.f);  // currently ignored
    torchstep->setPolarization(pol);
}


/*

Merged mesh 0 provides center_extent for all volumes in
global coordinates, the other merged mesh has only local to 
them center extent.::

    In [1]: ce0 = np.load("0/center_extent.npy")

    In [2]: ce0
    Out[2]: 
    array([[ -16520.   , -802110.   ,   -7125.   ,    7710.562],
           [ -16520.   , -802110.   ,    3892.9  ,   34569.875],
           [ -12840.846, -806876.25 ,    5389.855,   22545.562],
           ..., 
           [ -12195.957, -799312.625,   -7260.   ,    5000.   ],
           [ -17081.184, -794607.812,   -7260.   ,    5000.   ],
           [ -16519.908, -802110.   ,  -12410.   ,    7800.875]], dtype=float32)

    In [3]: ce0.shape
    Out[3]: (12230, 4)

For targetting the instances of merged meshes > 0 can use
the transform matrices::

    In [15]: tr1 = np.load("1/transforms.npy")

    In [17]: tr1.reshape(672,4,4)
    Out[17]: 
    array([[[      0.   ,      -0.   ,       1.   ,       0.   ],
            [      0.762,       0.648,       0.   ,       0.   ],
            [     -0.648,       0.762,       0.   ,       0.   ],
            [ -16572.902, -801469.625,   -8842.5  ,       1.   ]],



*/


void GGeo::dumpTree(const char* msg)
{
    GMergedMesh* mm0 = getMergedMesh(0);

    // all these are full traverse counts, not reduced by selections or instancing
    unsigned int nso = mm0->getNumSolids();  
    guint4* nodeinfo = mm0->getNodeInfo(); 
    unsigned int npv = m_pvlist->getNumItems(); 
    unsigned int nlv = m_lvlist->getNumItems(); 

    LOG(info) << msg 
              << " nso " << nso 
              << " npv " << npv 
              << " nlv " << nlv 
              << " nodeinfo " << (void*)nodeinfo
              ; 

    if( npv == 0 || nlv == 0 || nodeinfo == NULL )
    {
        LOG(warning) << "GGeo::dumpTree MISSING pvlist lvlist or nodeinfo " ; 
        return ;
    }
    else
    {
        assert(npv == nlv && nso == npv);
    }

    for(unsigned int i=0 ; i < nso ; i++)
    {
         guint4* info = nodeinfo + i ;  
         glm::ivec4 offnum = getNodeOffsetCount(i);
         std::string& pv = m_pvlist->getItem(i);
         std::string& lv = m_lvlist->getItem(i);
         printf(" %6u : nf %4d nv %4d id %6u pid %6d : %4d %4d %4d %4d  :%50s %50s \n", i, 
                    info->x, info->y, info->z, info->w,  offnum.x, offnum.y, offnum.z, offnum.w,
                    pv.c_str(), lv.c_str() ); 
    }
}


glm::ivec4 GGeo::getNodeOffsetCount(unsigned int index)
{
    GMergedMesh* mm0 = getMergedMesh(0);
    guint4* nodeinfo = mm0->getNodeInfo(); 
    unsigned int nso = mm0->getNumSolids();  
    assert(index < nso );

    glm::ivec4 offset ; 
    unsigned int cur_vert(0);
    unsigned int cur_face(0);

    for(unsigned int i=0 ; i < nso ; i++)
    {
        guint4* info = nodeinfo + i ;  
        if( i == index )
        {
           offset.x = cur_face ;   // cumulative sums of prior faces/verts in the buffer
           offset.y = cur_vert ;   //                  
           offset.z = info->x ;    // number faces/verts for this node
           offset.w = info->y ; 
           break ; 
        }
        cur_face += info->x ; 
        cur_vert += info->y ; 
    }
    return offset ; 
}


void GGeo::dumpVolume(unsigned int index, const char* msg)
{
    GMergedMesh* mm0 = getMergedMesh(0);
    unsigned int nsolid = mm0->getNumSolids();  
    unsigned int nvert = mm0->getNumVertices();  
    unsigned int nface = mm0->getNumFaces();  
    LOG(info) << msg 
              << " nsolid " << nsolid
              << " nvert" << nvert
              << " nface " << nface
               ; 

    glm::ivec4 offnum = getNodeOffsetCount(index);
    LOG(info) << " nodeoffsetcount " 
              << " index " << index
              << " x " << offnum.x
              << " y " << offnum.y
              << " z " << offnum.z
              << " w " << offnum.w
              ;

    gfloat3* verts = mm0->getVertices();
    guint3* faces = mm0->getFaces(); 

    for(unsigned int i=0 ; i < offnum.z ; i++)
    {
        guint3* f = faces + offnum.x + i ;    // offnum.x is cumulative sum of prior solid face counts

        //  GMergedMesh::traverse  already does vertex index offsetting corresponding to the other solid meshes incorporated in the merge
        gfloat3* v0 = verts + f->x ; 
        gfloat3* v1 = verts + f->y ; 
        gfloat3* v2 = verts + f->z ; 

        glm::vec3 p0(v0->x, v0->y, v0->z);
        glm::vec3 p1(v1->x, v1->y, v1->z);
        glm::vec3 p2(v2->x, v2->y, v2->z);
        glm::vec3 pc = (p0 + p1 + p2)/3.f ;
        glm::vec3 e0 = p1 - p0;
        glm::vec3 e1 = p0 - p2;
        glm::vec3 no = glm::normalize(glm::cross( e1, e0 ));

        printf(" i %3u f %4u %4u %4u : %10.3f %10.3f %10.3f    %10.3f %10.3f %10.3f     %10.3f %10.3f %10.3f   :  %10.3f %10.3f %10.3f \n", i, 
            f->x, f->y, f->z, 
            p0.x, p0.y, p0.z,
            p1.x, p1.y, p1.z,
            p2.x, p2.y, p2.z,
            no.x, no.y, no.z 
         ); 

    }
}


glm::vec4 GGeo::getFaceCenterExtent(unsigned int face_index, unsigned int solid_index, unsigned int mergedmesh_index )
{
   return getFaceRangeCenterExtent( face_index, face_index + 1 , solid_index, mergedmesh_index );
}

glm::vec4 GGeo::getFaceRangeCenterExtent(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mergedmesh_index )
{
    assert(mergedmesh_index == 0 && "instanced meshes not yet supported");
    GMergedMesh* mm = getMergedMesh(mergedmesh_index);
    assert(mm);
    unsigned int nsolid = mm->getNumSolids();  
    assert(solid_index < nsolid);

    glm::ivec4 offnum = getNodeOffsetCount(solid_index);
    gfloat3* verts = mm->getVertices();
    guint3* faces = mm->getFaces(); 

    assert(face_index0 <  offnum.z );  
    assert(face_index1 <= offnum.z );   // face_index1 needs to go 1 beyond

    glm::vec3 lo( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    glm::vec3 centroid ; 

    unsigned int nface = face_index1 - face_index0 ; 
    for(unsigned int face_index=face_index0 ; face_index < face_index1 ; face_index++)
    {

        guint3* f = faces + offnum.x + face_index ; // offnum.x is cumulative sum of prior solid face counts within the merged mesh
        gfloat3* v = NULL ; 

        gfloat3* v0 = verts + f->x ;
        gfloat3* v1 = verts + f->y ;
        gfloat3* v2 = verts + f->z ;

        glm::vec3 p0(v0->x, v0->y, v0->z);
        glm::vec3 p1(v1->x, v1->y, v1->z);
        glm::vec3 p2(v2->x, v2->y, v2->z);

        centroid = centroid + p0 + p1 + p2  ; 

        for(unsigned int i=0 ; i < 3 ; i++)
        {
            switch(i)
            {
                case 0: v = v0 ; break ; 
                case 1: v = v1 ; break ; 
                case 2: v = v2 ; break ; 
            }

            lo.x = std::min( lo.x, v->x);
            lo.y = std::min( lo.y, v->y);
            lo.z = std::min( lo.z, v->z);

            hi.x = std::max( hi.x, v->x);
            hi.y = std::max( hi.y, v->y);
            hi.z = std::max( hi.z, v->z);
        }
    }

    glm::vec3 dim = hi - lo ; 

    float extent = 0.f ;
    extent = std::max( dim.x , extent ); 
    extent = std::max( dim.y , extent ); 
    extent = std::max( dim.z , extent ); 
    extent = extent / 2.0f  ;

    glm::vec4 ce ; 
    if( nface == 1 )
    {
       // for single face using avg matches OpenGL geom shader, and OptiX
        ce.x = centroid.x/3.f ; 
        ce.y = centroid.y/3.f ; 
        ce.z = centroid.z/3.f ; 
    }
    else
    {
       // for multiple faces use bbox center, as there are repeated vertices
       ce.x = (hi.x + lo.x)/2.f ; 
       ce.y = (hi.y + lo.y)/2.f ; 
       ce.z = (hi.z + lo.z)/2.f ; 
    }
    ce.w = extent ; 
 
    return ce ; 
}
