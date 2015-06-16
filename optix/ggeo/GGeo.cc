#include "GGeo.hh"

#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GMaterial.hh"
#include "GSolid.hh"
#include "GMesh.hh"
#include "GSubstanceLib.hh"
#include "GMergedMesh.hh"

#include "assert.h"
#include "stdio.h"
#include "string.h"

#define BSIZ 50


void GGeo::init()
{

   m_substance_lib = new GSubstanceLib();

   // chroma/chroma/geometry.py
   // standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
   //
   GDomain<float>* standard_wavelengths = new GDomain<float>(60.f, 810.f, 20.f );  
   m_substance_lib->setStandardDomain( standard_wavelengths );
}


GGeo::~GGeo()
{
   delete m_low ; 
   delete m_high ; 
   delete m_substance_lib ;
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

// cannot do this check any more in GSubstance approach 

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



GMergedMesh* GGeo::getMergedMesh(unsigned int index)
{
    if(!m_merged_mesh)
    {
        m_merged_mesh = GMergedMesh::create(index, this);
    }
    return m_merged_mesh ;
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




