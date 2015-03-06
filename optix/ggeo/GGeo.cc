#include "GGeo.hh"

#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GMaterial.hh"
#include "GSolid.hh"
#include "GMesh.hh"

#include "assert.h"
#include "stdio.h"

#define BSIZ 50

GGeo::GGeo() :
   m_low(NULL),
   m_high(NULL)
{
   printf("GGeo::GGeo\n");
}

GGeo::~GGeo()
{
   delete m_low ; 
   delete m_high ; 
}

gfloat3* GGeo::getLow()
{
   return m_low ; 
}
gfloat3* GGeo::getHigh()
{
   return m_high ; 
}


void GGeo::setLow(const gfloat3& low)
{
    m_low = new gfloat3(low);
}
void GGeo::setHigh(const gfloat3& high)
{
    m_high = new gfloat3(high);
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
 
    /*

    char mbuf[BSIZ];
    for(unsigned int ims=0 ; ims < m_meshes.size()  ; ims++ )
    {
        GMesh* ms = m_meshes[ims];
        snprintf(mbuf,BSIZ, "%s ms %u", msg, ims);
        ms->Summary(mbuf);
    }

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
}

void GGeo::add(GMaterial* material)
{
    m_materials.push_back(material);
}

void GGeo::add(GBorderSurface* surface)
{
    m_border_surfaces.push_back(surface);
}

void GGeo::add(GSkinSurface* surface)
{
    m_skin_surfaces.push_back(surface);
}


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


unsigned int GGeo::getNumMeshes()
{
    return m_meshes.size();
}
unsigned int GGeo::getNumSolids()
{
    return m_solids.size();
}
unsigned int GGeo::getNumMaterials()
{
    return m_materials.size();
}
unsigned int GGeo::getNumBorderSurfaces()
{
    return m_border_surfaces.size();
}
unsigned int GGeo::getNumSkinSurfaces()
{
    return m_skin_surfaces.size();
}




GSolid* GGeo::getSolid(unsigned int index)
{
    return m_solids[index];
}
GSkinSurface* GGeo::getSkinSurface(unsigned int index)
{
    return m_skin_surfaces[index];
}
GBorderSurface* GGeo::getBorderSurface(unsigned int index)
{
    return m_border_surfaces[index];
}




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





