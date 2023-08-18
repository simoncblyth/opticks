// ./sn_accessor_test.sh 

#include "ssys.h"
#include "sn.h"
#include "s_csg.h"


void test_GetLVNodes()
{
    int lvid = ssys::getenvint("LVID", 108) ; 

    std::vector<sn*> nds ; 
    sn::GetLVNodes(nds, lvid) ; 

    std::cerr 
        << "test_GetLVNodes"
        << " lvid " << lvid 
        << std::endl 
        << sn::Desc(nds)
        << std::endl
        ;
}

void test_getLVNodes()
{
    int lvid = ssys::getenvint("LVID", 108) ; 
    sn* root = sn::GetLVRoot(lvid) ; 
    if(!root) return ; 

    std::vector<sn*> nds ; 
    root->getLVNodes(nds) ; 

    std::cerr 
        << "test_getLVNodes"
        << " lvid " << lvid 
        << std::endl 
        << sn::Desc(nds)
        << std::endl
        ;
}

void test_GetLVRoot()
{
    int lvid = ssys::getenvint("LVID", 108) ; 
    sn* root = sn::GetLVRoot(lvid) ; 

    std::cerr 
        << "test_GetLVRoot"
        << " lvid " << lvid 
        << std::endl 
        << ( root ? root->desc() : "-" )
        << std::endl
        <<  ( root ? root->render(sn::TYPETAG) : "-" )
        << std::endl
        <<  ( root ? root->rbrief() : "-" )
        << std::endl
        ;
}

void test_max_binary_depth()
{
    int lvid = ssys::getenvint("LVID", 108) ; 
    sn* root = sn::GetLVRoot(lvid) ; 

    std::cerr 
        << "test_max_binary_depth"
        << " lvid " << lvid 
        << std::endl 
        << ( root ? root->desc() : "-" )
        << std::endl
        <<  ( root ? root->render(sn::TYPETAG) : "-" )
        << std::endl
        <<  ( root ? root->rbrief() : "-" )
        << std::endl
        << " max_binary_depth " << ( root ? root->max_binary_depth() : -1 )
        << std::endl
        ;
}

void test_typenodes_()
{
    int lvid = ssys::getenvint("LVID", 108) ; 
    sn* root = sn::GetLVRoot(lvid) ; 

    std::vector<const sn*> nds0 ; 
    root->typenodes_(nds0, CSG_UNION );     

    std::vector<const sn*> nds1 ; 
    root->typenodes_(nds1, CSG_CYLINDER, CSG_CONE );     


    std::cerr
        << "test_typenodes_"
        << " lvid " << lvid 
        << std::endl 
        << ( root ? root->desc() : "-" )
        << std::endl
        <<  ( root ? root->render(sn::TYPETAG) : "-" )
        << std::endl
        <<  ( root ? root->rbrief() : "-" )
        << std::endl
        << " root->typenodes_(nds0, CSG_UNION) " 
        << std::endl
        << sn::Desc(nds0) 
        << std::endl 
        << " root->typenodes_(nds1, CSG_CYLINDER, CSG_CONE ) " 
        << std::endl
        << sn::Desc(nds1) 
        << std::endl 
        ;
}

void test_getLVNumNode()
{
    sn* root = sn::GetLVRoot(ssys::getenvint("LVID", 108)) ; 
    if(!root) return ; 

    std::cerr 
        << "test_getLVNumNode" 
        << std::endl 
        << root->desc()
        << std::endl 
        << " root->getLVNumNode() "
        << root->getLVNumNode() 
        << std::endl 
        << " root->getLVBinNode() "
        << root->getLVBinNode() 
        << std::endl 
        << " root->getLVSubNode() "
        << root->getLVSubNode() 
        << std::endl 
        ;
}


void test_GetLVNodesComplete()
{
    int lvid = ssys::getenvint("LVID", 108);

    std::vector<const sn*> nds ; 
    sn::GetLVNodesComplete(nds, lvid ) ; 

    std::cerr 
        << "test_GetLVNodesComplete" 
        << std::endl
        << sn::Desc(nds)
        << std::endl
        ;
}
void test_getLVNodesComplete()
{
    sn* root = sn::GetLVRoot(ssys::getenvint("LVID", 108)) ; 
    if(!root) return ; 

    std::vector<const sn*> nds ; 
    root->getLVNodesComplete(nds);     

    std::cerr 
        << "test_getLVNodesComplete" 
        << std::endl
        << sn::Desc(nds)
        << std::endl
        <<  ( root ? root->render(sn::TYPETAG) : "-" )
        << std::endl
        ;
}

void test_ancestors()
{
    sn* root = sn::GetLVRoot(ssys::getenvint("LVID", 108)) ; 
    if(!root) return ; 

    std::vector<const sn*> nds0 ; 
    root->ancestors(nds0);     
    assert( nds0.size() == 0 && "root nodes should not have ancestors" ); 

    std::cerr 
        << "test_ancestors" 
        << std::endl
        << sn::Desc(nds0)
        << std::endl
        <<  ( root ? root->render(sn::TYPETAG) : "-" )
        << std::endl
        ;


    std::vector<sn*> nds1 ; 
    root->getLVNodes(nds1); 

    std::cerr 
        << "test_ancestors" 
        << std::endl
        << " root->getLVNodes(nds1) "
        << std::endl
        << sn::Desc(nds1)
        << std::endl
        ;

   for(int i=0 ; i < int(nds1.size()) ; i++)
   {
       sn* nd = nds1[i] ; 
       std::vector<const sn*> anc ; 
       nd->ancestors(anc); 
       std::cerr
          << std::setw(3) << i << " : "
          << " anc.size " << anc.size()
          << std::endl 
          ;
   }   
}

void test_getNodeTransformProduct()
{
    sn* root = sn::GetLVRoot(ssys::getenvint("LVID", 108)) ; 
    if(!root) return ; 

    std::vector<sn*> nds ; 
    root->getLVNodes(nds); 

    std::cerr
        << "test_getNodeTransformProduct"
        << std::endl
        << sn::Desc(nds)
        << std::endl
        ;

    for(int i=0 ; i < int(nds.size()) ; i++)
    {
        sn* nd = nds[i] ; 

        glm::tmat4x4<double> t(1.); 
        glm::tmat4x4<double> v(1.); 
        bool reverse = false ; 

        std::cerr 
            << "test_getNodeTransformProduct"
            << std::endl
            << nd->desc_getNodeTransformProduct(t,v,reverse)
            << std::endl
            ;
    }
}

int main()
{
    s_csg::Load("$BASE");  

    /*
    test_GetLVNodes(); 
    test_getLVNodes(); 
    test_GetLVRoot(); 
    test_max_binary_depth(); 
    test_typenodes_(); 
    test_getLVNumNode(); 
    test_GetLVNodesComplete();     
    test_getLVNodesComplete();     
    test_ancestors();     
    */

    test_getNodeTransformProduct();     

    return 0 ; 
}
