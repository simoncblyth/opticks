#include <algorithm>
#include <iomanip>
#include <climits>

#include <boost/algorithm/string.hpp>
#include "BStr.hh"

// npy-
#include "NSensorList.hpp"
#include "NSensor.hpp"

//okc-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksQuery.hh"

#include "GVector.hh"
#include "GMatrix.hh"

#include "GDomain.hh" 
#include "GAry.hh" 
#include "GProperty.hh" 
#include "GPropertyMap.hh" 

#include "GMaterial.hh"
#include "GBorderSurface.hh"
#include "GSkinSurface.hh"
#include "GOpticalSurface.hh"

#include "GBndLib.hh"
#include "GSurfaceLib.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GGeo.hh"


#include "AssimpGGeo.hh"
#include "AssimpImporter.hh"
#include "AssimpTree.hh"
#include "AssimpSelection.hh"
#include "AssimpNode.hh"

#include <assimp/types.h>
#include <assimp/scene.h>



/*
        g4daeview.sh -g 3148:3155
        g4daeview.sh -g 4813:4816    Iws/SST/Oil  outside of SST high reflectivity 0.8, inside of SST low reflectivity 0.1
*/


// ctor macro facilitates GGeo staying ignorant of AssimpWrap and Assimp
#define GMATRIXF(m) \
           GMatrixF( \
                     (m).a1,(m).a2,(m).a3,(m).a4, \
                     (m).b1,(m).b2,(m).b3,(m).b4, \
                     (m).c1,(m).c2,(m).c3,(m).c4, \
                     (m).d1,(m).d2,(m).d3,(m).d4) \



#include "PLOG.hh"


AssimpGGeo::AssimpGGeo(GGeo* ggeo, AssimpTree* tree, AssimpSelection* selection) 
   : 
   m_ggeo(ggeo),
   m_tree(tree),
   m_selection(selection),
   m_domain_scale(1.f),
   m_values_scale(1.f),
   m_domain_reciprocal(true),
   m_skin_surface(0),
   m_inborder_surface(0),
   m_outborder_surface(0),
   m_no_surface(0),
   m_volnames(false),
   m_reverse(true),        // true: ascending wavelength ordering of properties
   m_cathode(NULL),
   m_verbosity(0)
{
    init();
}


bool AssimpGGeo::getVolNames()
{
    return m_volnames ; 
}
void AssimpGGeo::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}



void AssimpGGeo::init()
{
    // TODO: consolidate constant handling into okc-
    //       see also ggeo-/GConstant and probably elsewhere
    //
    // see g4daenode.py as_optical_property_vector

    double hc_over_GeV = 1.2398424468024265e-06 ;  // h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
    double hc_over_MeV = hc_over_GeV*1000. ;
    //float hc_over_eV  = hc_over_GeV*1.e9 ;

    m_domain_scale = static_cast<float>(hc_over_MeV) ; 
    m_values_scale = 1.0f ; 

    m_volnames = m_ggeo->isVolnames();
}



int AssimpGGeo::load(GGeo* ggeo)
{
    Opticks* opticks = ggeo->getOpticks();
    OpticksResource* resource = opticks->getResource();
    OpticksQuery* query = opticks->getQuery() ;

    const char* path = opticks->getDAEPath() ;
    const char* ctrl = resource->getCtrl() ;
    unsigned int verbosity = ggeo->getLoaderVerbosity();

    LOG(info)<< "AssimpGGeo::load "  
             << " path " << ( path ? path : "NULL" ) 
             << " query " << ( query ? query->getQueryString() : "NULL" )
             << " ctrl " << ( ctrl ? ctrl : "NULL" )
             << " verbosity " << verbosity 
             ; 

    assert(path);
    assert(query);
    assert(ctrl);

    AssimpImporter assimp(path);

    assimp.import();

    AssimpSelection* selection = assimp.select(query);

    AssimpTree* tree = assimp.getTree();


    AssimpGGeo agg(ggeo, tree, selection); 

    agg.setVerbosity(verbosity);

    int rc = agg.convert(ctrl);

    return rc ;
}


int AssimpGGeo::convert(const char* ctrl)
{
    LOG(info) << "AssimpGGeo::convert ctrl " << ctrl ; 

    const aiScene* scene = m_tree->getScene();

    convertMaterials(scene, m_ggeo, ctrl );
    convertSensors( m_ggeo ); 
    convertMeshes(scene, m_ggeo, ctrl);
    convertStructure(m_ggeo);

    return 0 ;
}

void AssimpGGeo::addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property )
{
    const char* shortname = pmap->getShortName();

    LOG(debug) << "AssimpGGeo::addPropertyVector " 
              << " shortname " << (shortname ? shortname : "-" )
              << " k " << k 
               ; 

    float* data = (float*)property->mData ;
    unsigned int nbyte  = property->mDataLength ; 
    unsigned int nfloat = nbyte/sizeof(float) ;
    assert(nfloat % 2 == 0 && nfloat > 1 );
    unsigned int npair  = nfloat/2 ;


    // dont scale placeholder -1 : 1 domain ranges
    double dscale = data[0] > 0 && data[npair-1] > 0 ? m_domain_scale : 1.f ;   
    double vscale = m_values_scale ; 

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

    std::vector<float> vals ; 
    std::vector<float> domain  ; 

    for( unsigned int i = 0 ; i < npair ; i++ ) 
    {
        double d0 = data[2*i] ; 
        double d = m_domain_reciprocal ? dscale/d0 : dscale*d0 ; 
        double v = data[2*i+1]*vscale  ;

        double dd = noscale ? d0 : d ; 

        domain.push_back( static_cast<float>(dd) );
        vals.push_back( static_cast<float>(v) );  

        //if( noscale && ( i < 5 || i > npair - 5) )
        //printf("%4d %10.3e %10.3e \n", i, domain.back(), vals.back() );
    }

    if(m_reverse)
    {
       std::reverse(vals.begin(), vals.end());
       std::reverse(domain.begin(), domain.end());
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



bool AssimpGGeo::hasVectorProperty(aiMaterial* material, const char* propname)
{
    aiMaterialProperty* prop = getVectorProperty(material, propname);
    return prop != NULL ; 
}

aiMaterialProperty* AssimpGGeo::getVectorProperty(aiMaterial* material, const char* propname )
{
    aiMaterialProperty* ret = NULL ;
    //unsigned int numProperties = material->mNumProperties ;
    for(unsigned int i = 0; i < material->mNumProperties; i++)
    {
        aiMaterialProperty* property = material->mProperties[i] ;
        aiString key = property->mKey ; 
        const char* k = key.C_Str();

        if( k[0] == '?' || k[0] == '$') continue ;   

        aiPropertyTypeInfo type = property->mType ; 
        if(type == aiPTI_Float && strncmp(k, propname, strlen(k))==0)
        { 
            ret = property ; 
            break ; 
        }
    }
    return ret ; 
}




void AssimpGGeo::addProperties(GPropertyMap<float>* pmap, aiMaterial* material )
{
    //unsigned int numProperties = material->mNumProperties ;
    for(unsigned int i = 0; i < material->mNumProperties; i++)
    {
        aiMaterialProperty* property = material->mProperties[i] ;
        aiString key = property->mKey ; 
        const char* k = key.C_Str();

        // skip Assimp standard material props $clr.emissive $mat.shininess ?mat.name  etc..
        if( k[0] == '?' || k[0] == '$') continue ;   

        //printf("AssimpGGeo::addProperties i %d k %s \n", i, k ); 

        aiPropertyTypeInfo type = property->mType ; 
        if(type == aiPTI_Float)
        { 
            addPropertyVector(pmap, k, property );
        }
        else if( type == aiPTI_String )
        {
            aiString val ; 
            material->Get(k,0,0,val);
            //const char* v = val.C_Str();
            //printf("skip k %s v %s \n", k, v ); needs props are plucked elsewhere
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
const char* AssimpGGeo::g4dae_skinsurface_volume        = "g4dae_skinsurface_volume" ;

const char* AssimpGGeo::g4dae_opticalsurface_name       = "g4dae_opticalsurface_name" ;
const char* AssimpGGeo::g4dae_opticalsurface_type       = "g4dae_opticalsurface_type" ;
const char* AssimpGGeo::g4dae_opticalsurface_model      = "g4dae_opticalsurface_model" ;
const char* AssimpGGeo::g4dae_opticalsurface_finish     = "g4dae_opticalsurface_finish" ;
const char* AssimpGGeo::g4dae_opticalsurface_value      = "g4dae_opticalsurface_value" ;

const char* AssimpGGeo::EFFICIENCY = "EFFICIENCY" ; 


void AssimpGGeo::convertMaterials(const aiScene* scene, GGeo* gg, const char* query )
{
    LOG(info)<<"AssimpGGeo::convertMaterials " 
             << " query " << query 
             << " mNumMaterials " << scene->mNumMaterials  
             ;

    //GDomain<float>* standard_domain = gg->getBoundaryLib()->getStandardDomain(); 
    GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain(); 


    for(unsigned int i = 0; i < scene->mNumMaterials; i++)
    {
        unsigned int index = i ;  // hmm, make 1-based later 

        aiMaterial* mat = scene->mMaterials[i] ;

        aiString name_;
        mat->Get(AI_MATKEY_NAME, name_);

        const char* name = name_.C_Str();

        //if(strncmp(query, name, strlen(query))!=0) continue ;  

        LOG(debug) << "AssimpGGeo::convertMaterials " << i << " " << name ;

        const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
        const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );

        const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );

        const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
        const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
        const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
        const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
        const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );


        GOpticalSurface* os = osnam && ostyp && osmod && osfin && osval ? new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) : NULL ; 


        // assimp "materials" are used to hold skinsurface and bordersurface properties, 
        // as well as material properties
        // which is which is determined by the properties present 

        if(os)
        {
            LOG(debug) << "AssimpGGeo::convertMaterials os " << i << " " << os->description(); 

            // assert(strcmp(osnam, name) == 0); 
            //      formerly enforced same-name convention between OpticalSurface 
            //      and the skin or border surface that references it, but JUNO doesnt follow that  
        }

        if( sslv )
        {
            assert(os && "all ss must have associated os");

            GSkinSurface* gss = new GSkinSurface(name, index, os);


            LOG(debug) << "AssimpGGeo::convertMaterials GSkinSurface " 
                      << " name " << name 
                      << " sslv " << sslv 
                      ; 

            gss->setStandardDomain(standard_domain);
            gss->setSkinSurface(sslv);
            addProperties(gss, mat );

            LOG(debug) << gss->description(); 
            gg->add(gss);

            {
                // without standard domain applied
                GSkinSurface*  gss_raw = new GSkinSurface(name, index, os);
                gss_raw->setSkinSurface(sslv);
                addProperties(gss_raw, mat );
                gg->addRaw(gss_raw);  // this was erroreously gss for a long time
            }   

        } 
        else if (bspv1 && bspv2 )
        {
            assert(os && "all bs must have associated os");
            GBorderSurface* gbs = new GBorderSurface(name, index, os);

            gbs->setStandardDomain(standard_domain);
            gbs->setBorderSurface(bspv1, bspv2);
            addProperties(gbs, mat );

            LOG(debug) << gbs->description(); 

            gg->add(gbs);

            {
                // without standard domain applied
                GBorderSurface* gbs_raw = new GBorderSurface(name, index, os);
                gbs_raw->setBorderSurface(bspv1, bspv2);
                addProperties(gbs_raw, mat );
                gg->addRaw(gbs_raw);
            }
        }
        else
        {
            assert(os==NULL);


            //printf("AssimpGGeo::convertMaterials aiScene materialIndex %u (GMaterial) name %s \n", i, name);
            GMaterial* gmat = new GMaterial(name, index);
            gmat->setStandardDomain(standard_domain);
            addProperties(gmat, mat );
            gg->add(gmat);

            {
                // without standard domain applied
                GMaterial* gmat_raw = new GMaterial(name, index);
                addProperties(gmat_raw, mat );
                gg->addRaw(gmat_raw);
            }

            if(hasVectorProperty(mat, EFFICIENCY ))
            {
                assert(gg->getCathode() == NULL && "only expecting one material with an EFFICIENCY property" );
                gg->setCathode(gmat) ;  
                m_cathode = mat ; 
            }

        }



        free((void*)bspv1);
        free((void*)bspv2);
        free((void*)sslv);

        free((void*)osnam);
        free((void*)ostyp);
        free((void*)osfin);
        free((void*)osmod);
        free((void*)osval);
    }


    m_ggeo->afterConvertMaterials(); 
}



void AssimpGGeo::convertSensors(GGeo* gg)
{
/*
Opticks is a surface based simulation, as opposed to 
Geant4 which is CSG volume based. In Geant4 hits are formed 
on stepping into volumes with associated SensDet.
The Opticks equivalent is intersecting with a "SensorSurface", 
which are fabricated by AssimpGGeo::convertSensors.
*/
    convertSensors( gg, m_tree->getRoot(), 0); 

    //assert(m_cathode);
    if(!m_cathode)
    {
         LOG(warning) << "AssimpGGeo::convertSensors m_cathode NULL : no material with an efficiency property ?  " ;
         return ; 
    }

    unsigned int nclv = gg->getNumCathodeLV();

    GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain(); 

    // DYB: nclv=2 for hemi and headon PMTs 
    for(unsigned int i=0 ; i < nclv ; i++)
    {
        const char* sslv = gg->getCathodeLV(i);
        LOG(info) << "AssimpGGeo::convertSensors" 
                  << " i " << i
                  << " sslv " << sslv 
                  ;

        std::string name = BStr::trimPointerSuffixPrefix(sslv, NULL );
        name += GSurfaceLib::SENSOR_SURFACE ; 

        const char* osnam = name.c_str() ;
        const char* ostyp = "0" ; 
        const char* osmod = "1" ; 
        const char* osfin = "3" ; 
        const char* osval = "1" ;  

        // TODO: check effects of above adhoc choice of common type/model/finish/value 
        // TODO: add parse ctor that understands: "type=dielectric_dielectric;model=unified;finish=ground;value=1.0"

        GOpticalSurface* os = new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) ; 

        // standard materials/surfaces use the originating aiMaterial index, 
        // extend that for fake SensorSurface by toting up all 

        unsigned int index = gg->getNumMaterials() + gg->getNumSkinSurfaces() + gg->getNumBorderSurfaces() ; 

        GSkinSurface* gss = new GSkinSurface(name.c_str(), index, os);

        gss->setStandardDomain(standard_domain);
       
        gss->setSkinSurface(sslv);

        gss->setSensor();
        // story continues in GBoundaryLib::standardizeSurfaceProperties

        addProperties(gss, m_cathode );

        LOG(info) << "AssimpGGeo::convertSensors gss " << gss->description(); 
 
        gg->add(gss);

        {
            // without standard domain applied
            GSkinSurface*  gss_raw = new GSkinSurface(name.c_str(), index, os);
            gss_raw->setSkinSurface(sslv);
            // not setting sensor, only the standardized need that
            addProperties(gss_raw, m_cathode );
            gg->addRaw(gss_raw);
        }   
    }
}

void AssimpGGeo::convertSensors(GGeo* gg, AssimpNode* node, unsigned int depth)
{
    // addCathodeLV into gg
    convertSensorsVisit(gg, node, depth);
    for(unsigned int i = 0; i < node->getNumChildren(); i++) convertSensors(gg, node->getChild(i), depth + 1);
}

void AssimpGGeo::convertSensorsVisit(GGeo* gg, AssimpNode* node, unsigned int depth)
{
    // collects lv of nodes of cathode material allowing construction 
    // of "fake" GSkinSurface
    //
    // NB border surface sensors at not handled, as there are non of those in DYB
    //

    unsigned int nodeIndex = node->getIndex();

    const char* lv   = node->getName(0); 

    const char* pv   = node->getName(1); 

    unsigned int mti = node->getMaterialIndex() ;
 
    GMaterial* mt = gg->getMaterial(mti);
    
    NSensorList* sens = gg->getSensorList();  
    /*
    NSensor* sensor0 = sens->getSensor( nodeIndex ); 
    NSensor* sensor1 = sens->findSensorForNode( nodeIndex ); 
    assert(sensor0 == sensor1);
    // these do not match
    */
    NSensor* sensor = sens->findSensorForNode( nodeIndex ); 

    if(sensor && mt == gg->getCathode())
    {
         LOG(debug) << "AssimpGGeo::convertSensorsVisit " 
                   << " depth " << depth 
                   << " lv " << lv  
                   << " pv " << pv  
                   ;
         gg->addCathodeLV(lv) ;   
    }
}



GMesh* AssimpGGeo::convertMesh(const aiMesh* mesh, unsigned int index )
{
    const char* meshname = mesh->mName.C_Str() ; 
    unsigned int numVertices = mesh->mNumVertices;
    unsigned int numFaces = mesh->mNumFaces;

    LOG(debug) << "AssimpGGeo::convertMesh " 
               << " index " << std::setw(4) << index
               << " v " << std::setw(4) << numVertices
               << " f " << std::setw(4) << numFaces
               << " n " << meshname 
               ; 


    aiVector3D* vertices = mesh->mVertices ; 
    gfloat3*   gvertices = new gfloat3[numVertices];
    for(unsigned int v = 0; v < numVertices; v++)
    {
        gvertices[v].x = vertices[v].x;
        gvertices[v].y = vertices[v].y;
        gvertices[v].z = vertices[v].z;
    }

    assert(mesh->HasNormals()); 
    aiVector3D* normals = mesh->mNormals ; 
    gfloat3* gnormals  = new gfloat3[numVertices];
    for(unsigned int v = 0; v < mesh->mNumVertices; v++)
    {
        gnormals[v].x = normals[v].x;
        gnormals[v].y = normals[v].y;
        gnormals[v].z = normals[v].z;
    }

    //aiFace* faces = mesh->mFaces ; 
    guint3*  gfaces = new guint3[numFaces];
    for(unsigned int f = 0; f < mesh->mNumFaces; f++)
    {
        aiFace face = mesh->mFaces[f];
        gfaces[f].x = face.mIndices[0];
        gfaces[f].y = face.mIndices[1];
        gfaces[f].z = face.mIndices[2];
    }

    gfloat2* gtexcoords = NULL ;

    GMesh* gmesh = new GMesh( index, gvertices, numVertices, gfaces, numFaces, gnormals, gtexcoords); 
    return gmesh ; 
}




unsigned AssimpGGeo::getNumMeshes()
{
    const aiScene* scene = m_tree->getScene();
    return scene->mNumMeshes ;
}

GMesh* AssimpGGeo::convertMesh(unsigned int index )
{
    const aiScene* scene = m_tree->getScene();
    assert(index < scene->mNumMeshes);
    aiMesh* mesh = scene->mMeshes[index] ;
    GMesh* graw = convertMesh(mesh, index );
    return graw ; 
}


void AssimpGGeo::convertMeshes(const aiScene* scene, GGeo* gg, const char* /*query*/)
{
    LOG(info)<< "AssimpGGeo::convertMeshes NumMeshes " << scene->mNumMeshes ;

    for(unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i] ;
    
        const char* meshname = mesh->mName.C_Str() ; 

        GMesh* graw = convertMesh(mesh, i );

        GMesh* gmesh = graw->makeDedupedCopy(); // removes duplicate vertices, re-indexing faces accordingly

        delete graw ; 

        gmesh->setName(meshname);

        GMesh* gfixed = gg->invokeMeshJoin(gmesh);  
 
        assert(gfixed) ; 

        if(gfixed != gmesh)
        {
            LOG(trace) << "AssimpGGeo::convertMeshes meshfixing was done for "
                        << " meshname " << meshname 
                        << " index " << i 
                         ; 

            delete gmesh ;
        }

        gg->add(gfixed);

    }
}



void AssimpGGeo::convertStructure(GGeo* gg)
{
    LOG(info) << "AssimpGGeo::convertStructure ";

    convertStructure(gg, m_tree->getRoot(), 0, NULL);

    LOG(info) << "AssimpGGeo::convertStructure found surfaces "
              << " skin " << m_skin_surface 
              << " outborder " << m_outborder_surface 
              << " inborder " << m_inborder_surface 
              << " no " << m_no_surface  ;

    //gg->reportMeshUsage("AssimpGGeo::convertStructure reportMeshUsage");

    if(m_selection)
    {

        LOG(info) << __func__ 
                  << " m_selection: " << m_selection
                  << " NumSelected " << m_selection->getNumSelected() 
                  ;
        aiVector3D* alow  = m_selection->getLow() ;
        gfloat3 low(alow->x, alow->y, alow->z);

        aiVector3D* ahigh = m_selection->getHigh() ;
        gfloat3 high(ahigh->x, ahigh->y, ahigh->z);

        gg->setLow(low);
        gg->setHigh(high);
    }

    //gg->Summary("AssimpGGeo::convertStructure");
}

void AssimpGGeo::convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
{
    // recursive traversal of the AssimpNode tree
    // note that full tree is traversed even when a partial selection is applied 


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




GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* /*parent*/)
{
    // Associates node to extra information analogous to collada_to_chroma.py:visit
    //
    // * outside/inside materials (parent/child assumption is expedient) 
    // * border surfaces, via pv pair names
    // * skin surfaces, via lv names
    //
    // Solid-centric naming 
    //
    // outer-surface 
    //      relevant to inwards going photons, from parent to self
    //
    // inner-surface
    //      relevant to outwards going photons, from self to parent  
    // 
    //
    // Skinsurface are not always leaves 
    // (UnstStainlessSteel cable trays with single child BPE inside). 
    // Nevetheless treat skinsurface like an outer border surface.
    //
    // NB sibling border surfaces are not handled, but there are none of these 
    //


    //AssimpNode* cnode = node->getChild(0);   // first child, if any
    AssimpNode* pnode = node->getParent();
    if(!pnode) pnode=node ; 

    unsigned int nodeIndex = node->getIndex();
    bool dbg = nodeIndex >= 3199 && nodeIndex <=3203 ; 

    aiMatrix4x4 g = node->getGlobalTransform();
    GMatrixF* gtransform = new GMATRIXF(g) ;

    aiMatrix4x4 l = node->getLevelTransform(-2); // -1 is always identity 
    GMatrixF* ltransform = new GMATRIXF(l) ; 
    unsigned int msi = node->getMeshIndex();
    GMesh* mesh = gg->getMesh(msi);

    if(!mesh)
    {
       LOG(fatal) << "AssimpGGeo::convertStructureVisit NULL mesh"
                  << " depth " << depth 
                  << " msi " << msi 
                  ;

    }
    assert(mesh);


    unsigned int mti = node->getMaterialIndex() ;
    GMaterial* mt = gg->getMaterial(mti);

    unsigned int mti_p = pnode->getMaterialIndex();
    GMaterial* mt_p = gg->getMaterial(mti_p);

    //printf("AssimpGGeo::convertStructureVisit nodeIndex %d (mti %u mt %p) (mti_p %u mt_p %p) (msi %u mesh %p) \n", nodeIndex, mti, mt, mti_p, mt_p,  msi, mesh  );
    if(dbg || m_verbosity > 1)
    LOG(info) << "AssimpGGeo::convertStructureVisit" 
               << " nodeIndex " << std::setw(6) << nodeIndex
               << " ( mti " << std::setw(4) << mti << " mt " << (void*)mt << " ) "
               << std::setw(20) << ( mt ? mt->getShortName() : "-" ) 
               << " ( mti_p " << std::setw(4) << mti_p << " mt_p " << (void*)mt_p << " ) " 
               << std::setw(20) << ( mt_p ? mt_p->getShortName() : "-" )
               << " ( msi " << std::setw(4) << msi << " mesh " << (void*)mesh << " ) " << ( mesh ? mesh->getName() : "-" )
               ;  


    if(m_verbosity > 1)
    gtransform->Summary("AssimpGGeo::convertStructureVisit gtransform");

    if(m_verbosity > 1)
    ltransform->Summary("AssimpGGeo::convertStructureVisit ltransform");


    GSolid* solid = new GSolid(nodeIndex, gtransform, mesh, UINT_MAX, NULL ); // sensor starts NULL
    solid->setLevelTransform(ltransform);

    const char* lv   = node->getName(0); 
    const char* pv   = node->getName(1); 
    const char* pv_p   = pnode->getName(1); 

    gg->countMeshUsage(msi, nodeIndex, lv, pv);

    GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
    GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
    GSkinSurface*   sks = gg->findSkinSurface(lv);          

    if(sks && sks->hasNameEnding("SensorSurface"))
    {
       LOG(debug) << "AssimpGGeo::convertStructureVisit"
                 << " sks " << sks->description()
                 ;  
    }

    LOG(debug) << __func__ 
              << " lv: " << lv
              << " pv: " << pv
              << " pv_p: " << pv_p
              << " obs: " << (obs?obs->getName():"")
              << " ibs: " << (ibs?ibs->getName():"")
              << " sks: " << (sks?sks->getName():"")
              ;
  
    unsigned int nsurf = 0 ;
    if(sks) nsurf++ ;
    if(ibs) nsurf++ ;
    if(obs) nsurf++ ;
    assert(nsurf == 0 || nsurf == 1 || nsurf == 2); 


    GPropertyMap<float>* isurf  = NULL ; 
    GPropertyMap<float>* osurf  = NULL ; 
    //GPropertyMap<float>* iextra = NULL ; 
    //GPropertyMap<float>* oextra = NULL ; 

    if(sks)
    {
        osurf = sks ; 
        if(m_skin_surface < 10)
            LOG(debug) << "AssimpGGeo::convertStructureVisit OSKIN " 
                      << std::setw(3) << m_skin_surface << " "
                      << osurf->description() ;  
        // TODO: surface census, see if inner skin makes any sense
        m_skin_surface++ ; 
    }
    else if(obs)
    {
        osurf = obs ; 
        LOG(debug) << "AssimpGGeo::convertStructureVisit OSURF " 
                  << std::setw(3) << m_outborder_surface << " "
                  << osurf->description() ;  

        m_outborder_surface++ ; 
    }
    else if(ibs)
    {
        isurf = ibs ; 
        LOG(debug) << "AssimpGGeo::convertStructureVisit ISURF " 
                   << std::setw(3) << m_inborder_surface << " "
                  << isurf->description() ;  

        m_inborder_surface++ ; 
    }
    else
    {
        m_no_surface++ ;
    }

    if(isurf && osurf) LOG(info) << "AssimpGGeo::convertStructureVisit boundary with both ISURF and OSURF defined " ;

    assert((isurf == NULL || osurf == NULL) && "tripwire to inform that both ISURF and OSURF are defined simultaneously" ) ;


    NSensorList* sens = gg->getSensorList();  
    NSensor* sensor = sens->findSensorForNode( nodeIndex ); 
    solid->setSensor( sensor );  

    GBndLib* blib = gg->getBndLib();  
    GSurfaceLib* slib = gg->getSurfaceLib();  

    // boundary identification via 4-uint 
    unsigned int boundary = blib->addBoundary( 
                                               mt_p->getShortName(),
                                               osurf ? osurf->getShortName() : NULL ,
                                               isurf ? isurf->getShortName() : NULL ,
                                               mt->getShortName()
                                             );

    solid->setBoundary(boundary);
    {
       // sensor indices are set even for non sensitive volumes in PMT viscinity
       // TODO: change that 
       // this is a workaround that requires an associated sensitive surface
       // in order for the index to be provided

        unsigned int surface = blib->getOuterSurface(boundary);
        bool oss = slib->isSensorSurface(surface); 
        unsigned int ssi = oss ? NSensor::RefIndex(sensor) : 0 ; 
        solid->setSensorSurfaceIndex( ssi ); 
    } 

    char* desc = node->getDescription("\n\noriginal node description"); 
    solid->setDescription(desc);
    solid->setName(node->getName());  // this is LV name, maybe set PV name too 

    if(m_volnames)
    {
        solid->setPVName(pv);
        solid->setLVName(lv);
    }



    free(desc);

    return solid ; 
}



/*

Why does this show up with isurf rather than osurf ?


GBoundaryLib boundary index 16 
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
property values are causing total of 73 different boundarys ...


GBoundaryLib boundary index 62 
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
GBoundaryLib boundary index 63 
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
