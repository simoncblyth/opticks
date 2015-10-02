#include "GCache.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

int main(int argc, char* argv[])
{
    GCache* m_cache = new GCache("GGEOVIEW_");

    GGeo* m_ggeo = new GGeo(m_cache);

    m_ggeo->loadFromCache();

    unsigned int nmm = m_ggeo->getNumMergedMesh();
    for(unsigned int i=1 ; i < nmm ; i++)
    { 
        GMergedMesh* mm = m_ggeo->getMergedMesh(i) ;
        unsigned int numSolids = mm->getNumSolids();
        unsigned int numSolidsSelected = mm->getNumSolidsSelected();

        LOG(info) << " i " << i 
                  << " numSolids " << numSolids       
                  << " numSolidsSelected " << numSolidsSelected ;      


        for(unsigned int j=0 ; j < numSolids ; j++)
        {
            gbbox bb = mm->getBBox(j);
            bb.Summary("bb");
        }

    }

    return 0 ;
}


