
#include "SSys.hh"
#include "BOpticksResource.hh"

#include "NGLTF.hpp"
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
       // scene->dumpAllInstances(mesh_idx);  // TODO: revive 
    }
}



struct NSceneTest 
{

    NSceneTest(const char* idpath)
        :
        _scene(NULL)
    {
        _bres.setupViaID(idpath); 
    }

    NSceneTest(const char* srcpath, const char* srcdigest)
        :
        _scene(NULL)
    {
        _bres.setupViaSrc(srcpath, srcdigest); 
    }

    void load()
    {
        const char* base = _bres.getGLTFBase();
        const char* name = _bres.getGLTFName();
        load(base, name);
    }

    void load(const char* base, const char* name)
    { 

        if(!NGLTF::Exists(base, name))
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

        NGeometry* source = new NGLTF(base, name, config_, scene_idx);
        _scene =  new NScene(source, idfold, dbgnode) ;
        //_scene = new NScene( base, name, idfold, config_, dbgnode, scene_idx  ); 

        assert(_scene);

        // _scene->dump("NSceneTest");  // TODO: revive
        //_scene->dumpNdTree();

        //scene->dumpAll();
        //test_makeInstanceTransformsBuffer(_scene);
    }

    BOpticksResource _bres ; 
    NScene*          _scene ;  

};


void test_ViaSrc()
{
    const char* srcpath = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    const char* srcdigest = SSys::getenvvar("DEBUG_OPTICKS_SRCDIGEST", "0123456789abcdef0123456789abcdef") ; 
    if(!srcpath) return ; 
    NSceneTest nst(srcpath, srcdigest);
    nst.load();
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* idpath = SSys::getenvvar("IDPATH");
    if(!idpath) return 0 ; 

    NSceneTest nst(idpath);
    nst.load();

    return 0 ; 
}
