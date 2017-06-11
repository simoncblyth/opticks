#include "NPolygonizer.hpp"

#include "NParameters.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPrism.hpp"
#include "NPart.hpp"

#include "NTrianglesNPY.hpp"
#include "NCSG.hpp"

#include "NMarchingCubesNPY.hpp"


#ifdef WITH_DualContouringSample
#include "NDualContouringSample.hpp"
#endif

#ifdef WITH_ImplicitMesher
#include "NImplicitMesher.hpp"
#endif

#include "NHybridMesher.hpp"



#include "PLOG.hh"


NPolyMode_t NPolygonizer::PolyMode(const char* poly)
{
    NPolyMode_t mode = POLY_NONE ; 

    if(     strcmp(poly, "MC")  == 0)  mode = POLY_MC ;
    else if(strcmp(poly, "DCS") == 0)  mode = POLY_DCS ;
    else if(strcmp(poly, "IM")  == 0)  mode = POLY_IM ; 
    else if(strcmp(poly, "HY")  == 0)  mode = POLY_HY ; 
    else if(strcmp(poly, "BSP") == 0)  mode = POLY_BSP ; 

    return mode ; 
}


const char* NPolygonizer::POLY_NONE_  = "POLY_NONE" ; 
const char* NPolygonizer::POLY_MC_  = "POLY_MC" ; 
const char* NPolygonizer::POLY_DCS_ = "POLY_DCS" ; 
const char* NPolygonizer::POLY_IM_  = "POLY_IM" ; 
const char* NPolygonizer::POLY_HY_  = "POLY_HY" ; 
const char* NPolygonizer::POLY_BSP_ = "POLY_BSP" ; 



const char* NPolygonizer::PolyModeString(NPolyMode_t polymode)
{
    const char* s = NULL ;
    switch( polymode)
    {
       case POLY_NONE : s = POLY_NONE_ ; break ; 
       case POLY_MC   : s = POLY_MC_ ; break ; 
       case POLY_DCS  : s = POLY_DCS_ ; break ; 
       case POLY_IM   : s = POLY_IM_ ; break ; 
       case POLY_HY   : s = POLY_HY_ ; break ; 
       case POLY_BSP  : s = POLY_BSP_ ; break ; 
    }
    return s ; 
}


NPolygonizer::NPolygonizer(NCSG* csg)
    :
    m_csg(csg), 
    m_root(csg->getRoot()),
    m_bbox(new nbbox(m_root->bbox())),
    m_meta(csg->getMetaParameters()),
    m_verbosity(m_meta->get<int>("verbosity", "0" )),
    m_index(m_csg->getIndex()),
    m_poly(NULL),
    m_polymode(POLY_NONE)
{
    assert(m_root);
    assert(m_meta);

    std::string poly = m_meta->get<std::string>("poly", "DCS");

    m_poly = strdup(poly.c_str());
    m_polymode = PolyMode(m_poly);

    std::string cfg = m_meta->get<std::string>("polycfg", "");
    m_polycfg = strdup(cfg.c_str());

    if(m_verbosity > 0)
    { 
        m_meta->dump("NPolygonizer::NPolygonizer(meta)");

        LOG(info) << "NPolygonizer::NPolygonizer"
                  << " poly " << m_poly 
                  << " polymode " << m_polymode
                  << " PolyModeString " << PolyModeString(m_polymode) 
                  << " polycfg " << m_polycfg 
                  ;
    }
}



NTrianglesNPY* NPolygonizer::polygonize()
{
    if(m_verbosity > 0)
    LOG(info) << "NPolygonizer::polygonize"
              << " treedir " << m_csg->getTreeDir()
              << " poly " << m_poly 
              << " polymode " << m_polymode 
              << " PolyModeString " << PolyModeString(m_polymode) 
              << " verbosity " << m_verbosity 
              << " polycfg " << m_polycfg 
              << " index " << m_index
              ;

    NTrianglesNPY* tris = NULL ; 

    switch( m_polymode )
    {
        case POLY_MC:  tris = marchingCubesNPY()    ; break ; 
        case POLY_DCS: tris = dualContouringSample(); break ; 
        case POLY_IM:  tris = implicitMesher()      ; break ;      
        case POLY_HY:  tris = hybridMesher()        ; break ;    
        case POLY_BSP: tris = hybridMesher()        ; break ;    
        default:   assert(0);
    }
    bool valid = checkTris(tris);

    if(!valid)
    {   
        if(m_verbosity > 0)
        LOG(warning) << "INVALID NPolygonizer tris with " << m_poly ; 
        delete tris ; 
        tris = NTrianglesNPY::box(*m_bbox);
        tris->setMessage("PLACEHOLDER");
    }   
    else
    {
        unsigned numTris = tris ? tris->getNumTriangles() : 0 ;
        LOG(info) << "NPolygonizer::polygonize OK " 
                  << " numTris " << numTris 
                  ; 
    }

    return tris ;
}


bool NPolygonizer::checkTris(NTrianglesNPY* tris)
{
    unsigned numTris = tris ? tris->getNumTriangles() : 0 ;

    nbbox* tris_bb = tris && numTris > 0 ? tris->findBBox() : NULL ;

    bool poly_valid = tris_bb ? m_bbox->contains(*tris_bb) : false  ;

    if(!poly_valid && m_verbosity > 0)
    {
        LOG(warning) << "NPolygonizer::checkTris INVALID POLYGONIZATION "
                     << " poly " << m_poly
                     << " index " << m_index 
                     << " numTris " << numTris
                     ;

        std::cout << " node_bb " << m_bbox->desc() << std::endl ; 
        std::cout << " tris_bb " << ( tris_bb ? tris_bb->desc() : "bb:NULL" ) << std::endl ; 
    }
    return poly_valid ;
}



NTrianglesNPY* NPolygonizer::marchingCubesNPY()
{
    int nx = m_meta->get<int>("nx", "15" );
    NMarchingCubesNPY poly(nx) ;
    NTrianglesNPY* tris = poly(m_root);
    return tris ; 
}

NTrianglesNPY* NPolygonizer::dualContouringSample()
{
    NTrianglesNPY* tris = NULL ; 
#ifdef WITH_DualContouringSample
    float threshold = m_meta->get<float>("threshold", "0.1" );
    int   nominal = m_meta->get<int>("nominal", "7" );  // 1 << 5 = 32, 1 << 6 = 64, 1 << 7 = 128  
    int   coarse  = m_meta->get<int>("coarse", "6" );  
    NDualContouringSample poly(nominal, coarse, m_verbosity, threshold ) ; 
    tris = poly(m_root);
#else
    assert(0 && "installation does not have DualContouringSample support" );
#endif
    return tris ;
}

NTrianglesNPY* NPolygonizer::implicitMesher()
{
    NTrianglesNPY* tris = NULL ; 
#ifdef WITH_ImplicitMesher
    int   resolution = m_meta->get<int>("resolution", "100" );
    int   ctrl = m_meta->get<int>("ctrl", "0" );
    float expand_bb = 1e-4 ; 
    std::string seeds = m_meta->get<std::string>("seeds", "" );
    NImplicitMesher poly(m_root, resolution, m_verbosity, expand_bb, ctrl, seeds ) ; 
    tris = poly();
#else
    assert(0 && "installation does not have ImplicitMesher support" );
#endif
    return tris ;
}


NTrianglesNPY* NPolygonizer::hybridMesher()
{
    if(m_verbosity > 0 )
    LOG(info) << "NPolygonizer::hybridMesher" ; 

    NTrianglesNPY* tris = NULL ; 

    const char* treedir = m_csg->getTreeDir();
    NHybridMesher poly(m_root, m_meta, treedir ) ;
 
    tris = poly();
    return tris ;
}


