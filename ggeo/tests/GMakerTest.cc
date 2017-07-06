//  ggv --gmaker

// sysrap-
#include "OpticksCSG.h"

#include "NGLM.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NSceneConfig.hpp"

#include "Opticks.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GMaker.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"

class GMakerTest 
{
   public:
       GMakerTest(Opticks* ok);
       void make();
       void makeFromCSG();
   private:
       Opticks* m_ok  ;
       GMaker*  m_maker ;  
};

GMakerTest::GMakerTest(Opticks* ok)
   :
   m_ok(ok),
   m_maker(new GMaker(ok))
{
}

void GMakerTest::make()
{
    glm::vec4 param(0.f,0.f,0.f,100.f) ; 

    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ; 

    GSolid* solid = m_maker->make(0u, CSG_SPHERE, param, spec );

    solid->Summary();

    const GMesh* mesh = solid->getMesh();

    mesh->dump();
}

void GMakerTest::makeFromCSG()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    nnode::Tests(nodes);

    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ; 

    unsigned verbosity = 1 ; 

    const NSceneConfig* config = new NSceneConfig(m_ok->getGLTFConfig()); 

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        n->dump();

        n->set_boundary(spec);

        NCSG* csg = NCSG::FromNode( n, config );

        csg->setMeta<std::string>("poly", "IM");

        GSolid* solid = m_maker->makeFromCSG(csg, verbosity );

        const GMesh* mesh = solid->getMesh();

        mesh->Summary();

        solid->Summary();
   }
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;
    GGEO_LOG__ ;

    Opticks ok(argc, argv);
    
    GMakerTest tst(&ok);

    tst.makeFromCSG();

}

