#include "GCache.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

int main(int argc, char* argv[])
{
    GCache gc("GGEOVIEW_");
    GGeo* gg = GGeo::load(gc.getIdPath());

    unsigned int nmm = gg->getNumMergedMesh();
    for(unsigned int i=1 ; i < nmm ; i++)
    { 
        GMergedMesh* mm = gg->getMergedMesh(i) ;
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


