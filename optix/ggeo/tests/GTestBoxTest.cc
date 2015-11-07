//  ggv --testbox

#include "GCache.hh"

int main(int argc, char** argv)
{
    GCache* m_cache = new GCache("GGEOVIEW_", "testbox.log", "info");
    m_cache->configure(argc, argv);

    return 1 ;
}
