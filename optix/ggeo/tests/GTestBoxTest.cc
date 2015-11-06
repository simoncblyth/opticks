//  ggv --testbox

#include "GCache.hh"
#include "GBndLib.hh"
#include "GMergedMesh.hh"
#include "GTestBox.hh"
#include "GPmt.hh"

int main(int argc, char** argv)
{
    GCache* m_cache = new GCache("GGEOVIEW_", "testbox.log", "info");
    m_cache->configure(argc, argv);

    GBndLib* m_bndlib = GBndLib::load(m_cache, true) ;

    GMergedMesh* mm = GMergedMesh::load(m_cache, 1);  // instance-1  triangulated PMT 5-solids
    mm->dumpSolids();

    gbbox bb = mm->getBBox(0);     // solid-0 contains them all

    GTestBox* m_testbox = NULL ;  

    ///////////////////////////////////////////////////////////

    GPmt* m_pmt = NULL ; 

    m_pmt = GPmt::load( m_cache, 0, NULL );

    m_pmt->dump();




    ////////// canonically done in GGeo::modifyGeometry //////////

    m_testbox = new GTestBox( m_cache) ;

    m_testbox->setBndLib(m_bndlib);

    m_testbox->configure(NULL);


    unsigned int mesh_index = 1000 ; 
    unsigned int node_index = mm->getNumSolids() ; // node indices need to be contiguous ?
    
    m_testbox->make(bb, mesh_index, node_index );
    m_testbox->dump("GGeo::modifyGeometry");

    GSolid* solid = m_testbox->getSolid();

    GMergedMesh* com = GMergedMesh::combine( mm->getIndex(), mm, solid );   

    com->Dump();
    com->setGeoCode('S');  // signal OGeo to use Analytic geometry

    // m_geolib->clear();
    // m_geolib->setMergedMesh( 0, cmm );
     


    return 1 ;
}
