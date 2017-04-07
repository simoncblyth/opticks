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


#include "PLOG.hh"

NPolygonizer::NPolygonizer(NCSG* csg)
    :
    m_csg(csg), 
    m_root(csg->getRoot()),
    m_bbox(new nbbox(m_root->bbox())),
    m_meta(csg->getMeta()),
    m_verbosity(m_meta->get<int>("verbosity", "1" )),
    m_index(m_csg->getIndex()),
    m_poly(NULL)
{
    assert(m_root);
    assert(m_meta);

    std::string poly = m_meta->get<std::string>("poly", "DCS");
    m_poly = strdup(poly.c_str());
}



NTrianglesNPY* NPolygonizer::polygonize()
{
    LOG(info) << "NPolygonizer::polygonize"
              << " poly " << m_poly 
              ;


    NTrianglesNPY* tris = NULL ; 

    if( strcmp(m_poly, "MC") == 0)
    {   
        tris = marchingCubesNPY();
    }   
    else if(strcmp(m_poly, "DCS") == 0)
    {   
        tris = dualContouringSample(); 
    }   
    else if(strcmp(m_poly, "IM") == 0)
    {
        tris = implicitMesher(); 
    }
    else
    {
        assert(0);
    }

    bool valid = checkTris(tris);

    if(!valid)
    {   
        LOG(warning) << "INVALID NPolygonizer tris with " << m_poly << "  triangles outside root bbox REPLACE WITH PLACEHOLDER " ;   
        delete tris ; 
        tris = NTrianglesNPY::box(*m_bbox);
    }   

    return tris ;
}


bool NPolygonizer::checkTris(NTrianglesNPY* tris)
{
    unsigned numTris = tris ? tris->getNumTriangles() : 0 ;

    nbbox* tris_bb = tris && numTris > 0 ? tris->findBBox() : NULL ;

    bool poly_valid = tris_bb ? m_bbox->contains(*tris_bb) : false  ;

    LOG(info) << "NPolygonizer::checkTris"
              << " poly " << m_poly
              << " index " << m_index 
              << " numTris " << numTris
              << " tris_bb " << ( tris_bb ? tris_bb->desc() : "bb:NULL" )
              << " poly_valid " << ( poly_valid ? "YES" : "NO" )
              ;   

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
    std::string seeds = m_meta->get<std::string>("seeds", "" );
    NImplicitMesher poly(m_root, resolution, m_verbosity, 1.01, ctrl, seeds ) ; 
    tris = poly();
#else
    assert(0 && "installation does not have ImplicitMesher support" );
#endif
    return tris ;
}


