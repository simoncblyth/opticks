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
    m_csg(csg)
{
}

NTrianglesNPY* NPolygonizer::polygonize()
{
    NCSG* csg = m_csg ; 

    nnode* root = csg->getRoot() ;

    assert(root);
    nbbox node_bb = root->bbox();
    NParameters* meta = csg->getMeta();
    assert(meta);

    std::string tessa = meta->get<std::string>("tessa", "DCS") ; 
    int   verbosity = meta->get<int>("verbosity", "1" );  
    unsigned index = csg->getIndex();

    NTrianglesNPY* tris = NULL ; 

    if( strcmp(tessa.c_str(), "MC") == 0)
    {   
        int nx = meta->get<int>("nx", "15" );
        NMarchingCubesNPY tessa(nx) ;
        tris = tessa(root);
    }   
    else if(strcmp(tessa.c_str(), "DCS") == 0)
    {   
#ifdef WITH_DualContouringSample
        float threshold = meta->get<float>("threshold", "0.1" );
        int   nominal = meta->get<int>("nominal", "7" );  // 1 << 5 = 32, 1 << 6 = 64, 1 << 7 = 128  
        int   coarse  = meta->get<int>("coarse", "6" );  
        NDualContouringSample tessa(nominal, coarse, verbosity, threshold ) ; 
        tris = tessa(root);
#else
       assert(0 && "installation does not have DualContouringSample support" );
#endif
    }   
    else if(strcmp(tessa.c_str(), "IM") == 0)
    {
#ifdef WITH_ImplicitMesher
        int   resolution = meta->get<int>("resolution", "100" );
        int   ctrl = meta->get<int>("ctrl", "0" );
        std::string seeds = meta->get<std::string>("seeds", "0,0,0" );
        NImplicitMesher tessa(resolution, verbosity, 1.01, ctrl, seeds ) ; 
        tris = tessa(root);
#else
       assert(0 && "installation does not have ImplicitMesher support" );
#endif
    }


    unsigned numTris = tris ? tris->getNumTriangles() : 0 ;

    nbbox* tris_bb = tris && numTris > 0 ? tris->findBBox() : NULL ;

    bool tessa_valid = tris_bb ? node_bb.contains(*tris_bb) : false  ;

    LOG(info) << "NPolygonizer::polygonize"
              << " tessa " << tessa
              << " index " << index 
              << " numTris " << numTris
              << " tris_bb " << ( tris_bb ? tris_bb->desc() : "bb:NULL" )
              << " tessa_valid " << ( tessa_valid ? "YES" : "NO" )
              ;   

     if(!tessa_valid)
     {   
         LOG(warning) << "INVALID Tesselation triangles outside node bbox REPLACE WITH PLACEHOLDER " ;   
         delete tris ; 
         tris = NTrianglesNPY::box(node_bb);
     }   

     return tris ;
}


