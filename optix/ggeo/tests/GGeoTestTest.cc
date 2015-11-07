//  ggv --geotest

#include "GCache.hh"
#include "GGeoTest.hh"

int main(int argc, char** argv)
{
    GCache* m_cache = new GCache("GGEOVIEW_", "geotest.log", "info");
    m_cache->configure(argc, argv);

    GGeoTest* m_geotest = new GGeoTest(m_cache);
    m_geotest->configure();

    m_geotest->modifyGeometry();



    return 1 ;
}
