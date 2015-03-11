#include "AssimpGGeo.hh"
#include "AssimpTree.hh"
#include "AssimpSelection.hh"
#include "AssimpNode.hh"

#include <assimp/types.h>
#include <assimp/scene.h>

#include "GVector.hh"
#include "GMatrix.hh"
#include "GGeo.hh"
#include "GMesh.hh"
#include "GMaterial.hh"
#include "GBorderSurface.hh"
#include "GSkinSurface.hh"
#include "GSolid.hh"
#include "GSubstance.hh"
#include "GSubstanceLib.hh"

/*
        g4daeview.sh -g 3148:3155
        g4daeview.sh -g 4813:4816    Iws/SST/Oil  outside of SST high reflectivity 0.8, inside of SST low reflectivity 0.1
*/


AssimpGGeo::AssimpGGeo(AssimpTree* tree, AssimpSelection* selection) 
   : 
   m_tree(tree),
   m_selection(selection),
   m_domain_scale(1.f),
   m_values_scale(1.f),
   m_domain_reciprocal(true),
   m_inborder_surface(0),
   m_outborder_surface(0),
   m_skin_surface(0),
   m_no_surface(0)
{
    // see g4daenode.py as_optical_property_vector

    float hc_over_GeV = 1.2398424468024265e-06 ;  // h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
    float hc_over_MeV = hc_over_GeV*1000. ;
    //float hc_over_eV  = hc_over_GeV*1.e9 ;

    m_domain_scale = hc_over_MeV ; 
    m_values_scale = 1.0f ; 
}

AssimpGGeo::~AssimpGGeo()
{
}


GGeo* AssimpGGeo::convert(const char* ctrl)
{
    printf("AssimpGGeo::convert ctrl %s \n",ctrl);
    GGeo* gg = new GGeo();
    const aiScene* scene = m_tree->getScene();
    convertMaterials(scene, gg, ctrl);
    convertMeshes(scene, gg, ctrl);
    convertStructure(gg);

#if 0
    gg->materialConsistencyCheck();
#endif
    return gg ;
}

void AssimpGGeo::addPropertyVector(GPropertyMap* pmap, const char* k, aiMaterialProperty* property )
{
    unsigned int nbyte  = property->mDataLength ; 
    unsigned int nfloat = nbyte/sizeof(float) ;
    assert(nfloat % 2 == 0 && nfloat > 1 );
    unsigned int npair  = nfloat/2 ;


    std::vector<float> vals ; 
    std::vector<float> domain  ; 

    float* data = (float*)property->mData ;

    // dont scale placeholder -1 : 1 domain ranges
    float dscale = data[0] > 0 && data[npair-1] > 0 ? m_domain_scale : 1.f ;   
    float vscale = m_values_scale ; 


    // debug some funny domains : default zeros coming from somewhere 
    bool noscale =     ( pmap->isSkinSurface() 
                            && 
                         ( 
                            strcmp(k, "RINDEX") == 0 
                         )
                       )
                    || 
                       ( pmap->isBorderSurface() 
                             && 
                         ( 
                            strcmp(k, "BACKSCATTERCONSTANT") == 0 
                            ||
                            strcmp(k, "SPECULARSPIKECONSTANT") == 0 
                         )
                       ) ;   


    //if(noscale) 
    //    printf("AssimpGGeo::addPropertyVector k %-35s nbyte %4u nfloat %4u npair %4u \n", k, nbyte, nfloat, npair);

    for(unsigned int i=0 ; i < npair ; i++)
    {
        float d0 = data[2*i] ; 
        float d = m_domain_reciprocal ? dscale/d0 : dscale*d0 ; 

        domain.push_back( noscale ? d0 : d );
        vals.push_back(   data[2*i+1]*vscale  );

        //if( noscale && ( i < 5 || i > npair - 5) )
        //printf("%4d %10.3e %10.3e \n", i, domain.back(), vals.back() );
    }
    pmap->addProperty(k, vals.data(), domain.data(), vals.size() );
}




const char* AssimpGGeo::getStringProperty(aiMaterial* material, const char* query)
{
    for(unsigned int i = 0; i < material->mNumProperties; i++)
    {
        aiMaterialProperty* property = material->mProperties[i] ;
        aiString key = property->mKey ; 
        const char* k = key.C_Str();

        // skip Assimp standard material props $clr.emissive $mat.shininess ?mat.name  etc..
        if( k[0] == '?' || k[0] == '$') continue ;   

        aiPropertyTypeInfo type = property->mType ; 
        if(type == aiPTI_String)
        {
           aiString val ; 
           material->Get(k,0,0,val);
           const char* v = val.C_Str();
           if(strncmp(k, query, strlen(query))==0 ) return strdup(v) ;
        }
    }
    return NULL ;
}


void AssimpGGeo::addProperties(GPropertyMap* pmap, aiMaterial* material)
{
    //
    // TODO: pull this out into top level API
    //  chroma/chroma/geometry.py
    //  standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
    //
    float low(60.f);
    float high(810.f);
    float step(20.f);

    pmap->setStandardDomain(low,high,step); 

    unsigned int numProperties = material->mNumProperties ;
    for(unsigned int i = 0; i < material->mNumProperties; i++)
    {
        aiMaterialProperty* property = material->mProperties[i] ;
        aiString key = property->mKey ; 
        const char* k = key.C_Str();

        // skip Assimp standard material props $clr.emissive $mat.shininess ?mat.name  etc..
        if( k[0] == '?' || k[0] == '$') continue ;   

        aiPropertyTypeInfo type = property->mType ; 
        if(type == aiPTI_Float)
        { 
            addPropertyVector(pmap, k, property);
        }
        else if( type == aiPTI_String )
        {
            aiString val ; 
            material->Get(k,0,0,val);
            const char* v = val.C_Str();
            //printf("skip k %s v %s \n", k, v );
        }
        else
        {
            printf("skip k %s \n", k);
        }
    }
    //printf("addProperties props %2d %s \n", numProperties, pmap->getName());
}

void AssimpGGeo::setDomainScale(float domain_scale)
{
    m_domain_scale = domain_scale ; 
}
void AssimpGGeo::setValuesScale(float values_scale)
{
    m_values_scale = values_scale  ; 
}




const char* AssimpGGeo::g4dae_bordersurface_physvolume1 = "g4dae_bordersurface_physvolume1" ;
const char* AssimpGGeo::g4dae_bordersurface_physvolume2 = "g4dae_bordersurface_physvolume2" ;
const char* AssimpGGeo::g4dae_skinsurface_volume = "g4dae_skinsurface_volume" ;

void AssimpGGeo::convertMaterials(const aiScene* scene, GGeo* gg, const char* query)
{
    for(unsigned int i = 0; i < scene->mNumMaterials; i++)
    {
        aiMaterial* mat = scene->mMaterials[i] ;
        aiString name_;
        mat->Get(AI_MATKEY_NAME, name_);

        const char* name = name_.C_Str();

        if(strncmp(query, name, strlen(query))!=0) continue ;  

        const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
        const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );
        const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );

        if( sslv )
        {
            printf("AssimpGGeo::convertMaterials materialIndex %u sslv %s  \n", i, sslv);
            GSkinSurface*  gss = new GSkinSurface(name, i);
            gss->setSkinSurface(sslv);
            addProperties(gss, mat);
            gg->add(gss);
        } 
        else if (bspv1 && bspv2 )
        {
            printf("AssimpGGeo::convertMaterials materialIndex %u\n    bspv1 %s\n    bspv2 %s \n", i, bspv1, bspv2 );
            GBorderSurface* gbs = new GBorderSurface(name, i);
            gbs->setBorderSurface(bspv1, bspv2);
            addProperties(gbs, mat);
            gg->add(gbs);
        }
        else
        {
            printf("AssimpGGeo::convertMaterials materialIndex %u mt %s \n", i, name);
            GMaterial* gmat = new GMaterial(name, i);
            addProperties(gmat, mat);
            gg->add(gmat);
        }

        free((void*)bspv1);
        free((void*)bspv2);
        free((void*)sslv);
    }
}


void AssimpGGeo::convertMeshes(const aiScene* scene, GGeo* gg, const char* query)
{
    for(unsigned int i = 0; i < scene->mNumMeshes; i++)
    {

        aiMesh* mesh = scene->mMeshes[i] ;
        unsigned int numVertices = mesh->mNumVertices;

        aiVector3D* vertices = mesh->mVertices ; 
        gfloat3* gvertices = new gfloat3[numVertices];

        for(unsigned int v = 0; v < mesh->mNumVertices; v++)
        {
            gvertices[v].x = vertices[v].x;
            gvertices[v].y = vertices[v].y;
            gvertices[v].z = vertices[v].z;
        }


        unsigned int numFaces = mesh->mNumFaces;
        aiFace* faces = mesh->mFaces ; 
        guint3*  gfaces = new guint3[numFaces];

        for(unsigned int f = 0; f < mesh->mNumFaces; f++)
        {
            aiFace face = mesh->mFaces[f];
            gfaces[f].x = face.mIndices[0];
            gfaces[f].y = face.mIndices[1];
            gfaces[f].z = face.mIndices[2];
        }


        GMesh* gmesh = new GMesh( i, gvertices, numVertices, gfaces, numFaces ); 
        gg->add(gmesh);
    }
}


void AssimpGGeo::convertStructure(GGeo* gg)
{
    printf("AssimpGGeo::convertStructure\n");
    convertStructure(gg, m_tree->getRoot(), 0, NULL);

    if(m_selection)
    {

        aiVector3D* alow  = m_selection->getLow() ;
        gfloat3 low(alow->x, alow->y, alow->z);

        aiVector3D* ahigh = m_selection->getHigh() ;
        gfloat3 high(ahigh->x, ahigh->y, ahigh->z);

        gg->setLow(low);
        gg->setHigh(high);
    }

    gg->Summary("AssimpGGeo::convertStructure");
}

void AssimpGGeo::convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
{
    GSolid* solid = convertStructureVisit( gg, node, depth, parent);

    bool selected = m_selection && m_selection->contains(node) ;  

    solid->setSelected(selected);

    gg->add(solid);

    if(parent) // GNode hookup
    {
        parent->addChild(solid);
        solid->setParent(parent);
    }
    else
    {
        assert(node->getIndex() == 0);   // only root node has no parent 
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) convertStructure(gg, node->getChild(i), depth + 1, solid);
}

GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
{
    // Associations node to extra information analogous to collada_to_chroma.py:visit
    //
    // * outside/inside materials (parent/child assumption is expedient) 
    // * border surfaces, via pv pair names
    // * skin surfaces, via lv names
    //
    // Solid-centric naming 
    //
    // outer-surface 
    //      corresponds to inwards going photons, from parent to self
    //
    // inner-surface
    //       corresponds to outwards going photons, from self to parent  
    // 
    //
    // Skinsurface are not always leaves 
    // (UnstStainlessSteel cable trays with single child BPE inside). 
    // Nevetheless treat skinsurface like an outer border surface.
    //
    // NB sibling border surfaces are not handled, but there are none of these 
    //


    AssimpNode* cnode = node->getChild(0);   // first child, if any
    AssimpNode* pnode = node->getParent();
    if(!pnode) pnode=node ; 

    unsigned int nodeIndex = node->getIndex();

    aiMatrix4x4 m = node->getGlobalTransform();
    GMatrixF* transform = new GMatrixF(
                     m.a1,m.a2,m.a3,m.a4,  
                     m.b1,m.b2,m.b3,m.b4,  
                     m.c1,m.c2,m.c3,m.c4,  
                     m.d1,m.d2,m.d3,m.d4);

    unsigned int msi = node->getMeshIndex();
    GMesh* mesh = gg->getMesh(msi);

    unsigned int mti = node->getMaterialIndex() ;
    GMaterial* mt = gg->getMaterial(mti);

    unsigned int mti_p = pnode->getMaterialIndex();
    GMaterial* mt_p = gg->getMaterial(mti_p);

    GSolid* solid = new GSolid(nodeIndex, transform, mesh, NULL );

    const char* lv   = node->getName(0); 
    const char* pv   = node->getName(1); 
    const char* pv_p   = pnode->getName(1); 

    GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
    GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
    GSkinSurface*   sks = gg->findSkinSurface(lv);          
   
    unsigned int nsurf = 0 ;
    if(sks) nsurf++ ;
    if(ibs) nsurf++ ;
    if(obs) nsurf++ ;
    assert(nsurf == 0 || nsurf == 1 ); 


    GPropertyMap* osurf = NULL ; 
    GPropertyMap* isurf = NULL ; 

    if(sks)
    {
        m_skin_surface++ ; 
        osurf = sks ; 
    }
    else if(obs)
    {
        m_outborder_surface++ ; 
        osurf = obs ; 
    }
    else if(ibs)
    {
        m_inborder_surface++ ; 
        isurf = ibs ; 
    }
    else
    {
        m_no_surface++ ;
    }


    GSubstanceLib* lib = gg->getSubstanceLib();
    GSubstance* substance = lib->get(mt, mt_p, isurf, osurf ); 
    //substance->Summary("subst");

    solid->setSubstance(substance);  



    char* desc = node->getDescription("\n\noriginal node description"); 
    solid->setDescription(desc);
    free(desc);

    return solid ; 
}



/*

Why does this show up with isurf rather than osurf ?


GSubstanceLib substance index 16 
imat material 59 __dd__Materials__MineralOil0xbf5c830
ABSLENGTH
   0    899.871    219.400
   1    898.892    236.700
   2    897.916    257.300
   3    896.877    278.000
   4    895.905    292.700
 539    190.977     10.800
 540    189.976     11.100
 541    120.023     11.100
 542     79.990     11.100
RAYLEIGH
   0    799.898 500000.000
   1    699.922 300000.000
   2    589.839 170000.000
   3    549.819 100000.000
   4    489.863  62000.000
   7    299.986   7600.000
   8    199.975    850.000
   9    120.023    850.000
  10     79.990    850.000
RINDEX
   0    799.898      1.456
   1    690.701      1.458
   2    589.002      1.462
   3    546.001      1.464
   4    486.001      1.468
  14    139.984      1.642
  15    129.990      1.534
  16    120.023      1.434
  17     79.990      1.434
isurf bordersurface 4 __dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface
REFLECTIVITY
   0    799.898      0.100
   1    199.975      0.100
   2    120.023      0.100
   3     79.990      0.100




Many instances of skin surfaces with differing names but the same 
property values are causing total of 73 different substances ...


GSubstanceLib substance index 62 
imat material 75 __dd__Materials__UnstStainlessSteel0xc5c11e8
ABSLENGTH
   0    799.898      0.001
   1    199.975      0.001
   2    120.023      0.001
   3     79.990      0.001
osurf skinsurface 38 __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib5Surface
REFLECTIVITY
   0    826.562      0.400
   1    190.745      0.400
RINDEX
   0      0.000      0.000
   1      0.000      0.000
GSubstanceLib substance index 63 
imat material 75 __dd__Materials__UnstStainlessSteel0xc5c11e8
ABSLENGTH
   0    799.898      0.001
   1    199.975      0.001
   2    120.023      0.001
   3     79.990      0.001
osurf skinsurface 37 __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib4Surface
REFLECTIVITY
   0    826.562      0.400
   1    190.745      0.400
RINDEX
   0      0.000      0.000
   1      0.000      0.000



Adjust identity to be based on a property name and hash alone ?



*/
