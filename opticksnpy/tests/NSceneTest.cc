
#include "SSys.hh"
#include "BOpticksResource.hh"

#include "NScene.hpp"
#include "NSceneConfig.hpp"
#include "NPY.hpp"
#include "NPY_LOG.hh"

#include "PLOG.hh"


void test_makeInstanceTransformsBuffer(NScene* scene)
{
    unsigned num_mesh = scene->getNumMeshes();
    for(unsigned mesh_idx=0 ; mesh_idx < num_mesh ; mesh_idx++)
    {    
        NPY<float>* buf = scene->makeInstanceTransformsBuffer(mesh_idx);
        std::cout 
           << " mesh_idx " << std::setw(3) << mesh_idx 
           << " instance transforms " << std::setw(3) << buf->getNumItems()
           << std::endl ;  
        buf->dump();
        scene->dumpAllInstances(mesh_idx); 
    }
}



struct NSceneTest 
{
    NSceneTest(const char* srcpath, const char* srcdigest)
        :
        _scene(NULL)
    {
        _bres.setSrcPathDigest(srcpath, srcdigest); 
    }

    void load()
    {
        const char* base = _bres.getGLTFBase();
        const char* name = _bres.getGLTFName();
        load(base, name);
    }

    void load(const char* base, const char* name)
    { 

        if(!NScene::Exists(base, name))
        {
            LOG(warning) << "no such scene at"
                         << " base " << base
                         << " name " << name
                         ;
            return ; 
        } 

        const char* config = NULL ; 

        int dbgnode = -1 ; 
        int scene_idx = 0 ; 
        const char* idfold = NULL ;  

        NSceneConfig* config_ = new NSceneConfig(config);

        _scene = new NScene( base, name, idfold, config_, dbgnode, scene_idx  ); 
        assert(_scene);

        _scene->dump("NSceneTest");
        //_scene->dumpNdTree();

        //scene->dumpAll();
        //test_makeInstanceTransformsBuffer(_scene);
    }

    BOpticksResource _bres ; 
    NScene*          _scene ;  

};



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* srcpath = SSys::getenvvar("OPTICKS_SRCPATH");
    const char* srcdigest = "dummy" ; 
    if(!srcpath) return 0 ; 

    NSceneTest nst(srcpath, srcdigest);
    nst.load();

    return 0 ; 
}
