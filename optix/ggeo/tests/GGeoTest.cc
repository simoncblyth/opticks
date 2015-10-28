#include "GCache.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal




void misc(GGeo* m_ggeo)
{
    unsigned int nmm = m_ggeo->getNumMergedMesh();
    for(unsigned int i=0 ; i < nmm ; i++)
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


        GBuffer* friid = mm->getFaceRepeatedInstancedIdentityBuffer();
        if(friid) friid->save<unsigned int>("/tmp/friid.npy");

        GBuffer* frid = mm->getFaceRepeatedIdentityBuffer();
        if(frid) frid->save<unsigned int>("/tmp/frid.npy");

    }
}



int main(int argc, char* argv[])
{
    GCache* m_cache = new GCache("GGEOVIEW_");

    GGeo* m_ggeo = new GGeo(m_cache);

    m_ggeo->loadFromCache();


    m_ggeo->dumpStats();


    return 0 ;
}


