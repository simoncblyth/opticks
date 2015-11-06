//  ggv --testbox

#include "GCache.hh"
#include "GBndLib.hh"
#include "GMergedMesh.hh"
#include "GTestBox.hh"
#include "GPmt.hh"

struct NSlice ; 


int main(int argc, char** argv)
{
    GCache* m_cache = new GCache("GGEOVIEW_", "testbox.log", "info");
    m_cache->configure(argc, argv);

    GBndLib* m_bndlib = GBndLib::load(m_cache, true) ;

    GMergedMesh* mm = GMergedMesh::load(m_cache, 1);  // instance-1  triangulated PMT 5-solids
    mm->dumpSolids();

    //// mock GGeo context 

    GTestBox* m_testbox = NULL ;  
    GPmt* m_pmt = NULL ; 
    GMergedMesh* m_dynamic = NULL ; 

    ////////// below canonically done in GGeo::modifyGeometry //////////

    gbbox bb = mm->getBBox(0);     // solid-0 contains them all
    bb.enlarge(3.f);               // the **ONE** place for sizing containment box

    m_pmt = GPmt::load( m_cache, 0, NULL );  // part slicing disfavored, as only works at one level 
    m_pmt->dump();
    assert( m_pmt->getNumSolids() == mm->getNumSolids() );

    // prep the extra solid
    m_testbox = new GTestBox(m_cache) ;
    m_testbox->setBBox(bb);
    m_testbox->setBndLib(m_bndlib);
    m_testbox->configure(NULL);
    m_testbox->make(1000, mm->getNumSolids() ); // mesh_index, node_index: node indices need to be contiguous  
    m_testbox->dump("GGeo::modifyGeometry");

    GSolid* solid = m_testbox->getSolid();

    // create combined mesh 
    m_dynamic = GMergedMesh::combine( mm->getIndex(), mm, solid );   
    m_dynamic->Dump();
    m_dynamic->setGeoCode('S');  // signal OGeo to use Analytic geometry, TODO: set it not signal it  

    unsigned int nodeindex = m_pmt->getNumSolids();
    m_pmt->addContainer(bb, nodeindex );
    m_pmt->dump("after addContainer");

    assert( m_dynamic->getNumSolids() == m_pmt->getNumSolids() );
    

    // m_geolib->clear();
    // m_geolib->setMergedMesh( 0, m_dynamic );
     


    return 1 ;
}
