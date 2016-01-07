/*
   For debugging torch configuration..
   this test has to live in ggeo- rather than npy- 
   as ggeo- info is required for targetting


   ggv --torchstep  "frame=3201;source=0,0,1000;target=0,0,0;radius=300;"   "frame=3153;source=0,0,1000;target=0,0,0;radius=300;" 
*/

#include <cstring>

#include "GCache.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include "Opticks.hh"

//npy-
#include "TorchStepNPY.hpp"
#include "NLog.hpp"


int main(int argc, char* argv[])
{
    Opticks* opticks = new Opticks(argc, argv, "torch.log");
    GCache* m_cache = new GCache(opticks);

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


    unsigned int TORCH = 0x1 << 12 ; // copy of enum value from cu/photon.h

    TorchStepNPY* m_torchstep = new TorchStepNPY(TORCH, 1);


    if(argc > 1)
    {
       for(unsigned int i=1 ; i < argc ; i++)
       {
            const char* arg = argv[i] ;

            if(strcmp(arg,"--torchstep")==0) continue ;

            LOG(info) << "arg " << arg ; 

            m_torchstep->configure(arg);

            m_ggeo->targetTorchStep(m_torchstep);

            bool verbose=true ; 
            m_torchstep->addStep(verbose) ;

       }
    }



    return 0 ;
}


