// ./CSGNodeTest.sh 

#include <bitset>
#include <vector>
#include <iomanip>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "scuda.h"
#include "CSGNode.h"
#include "Sys.h"

#include "SBitSet.hh"
#include "CSGFoundry.h"
#include "CSGCopy.h"

#include "OPTICKS_LOG.hh"

void test_copy()
{
    LOG(info); 
    glm::mat4 m0(1.f); 
    glm::mat4 m1(2.f); 
    glm::mat4 m2(3.f); 

    m0[0][3] = Sys::int_as_float(42); 
    m1[0][3] = Sys::int_as_float(52); 
    m2[0][3] = Sys::int_as_float(62); 

    std::vector<glm::mat4> node ; 
    node.push_back(m0); 
    node.push_back(m1); 
    node.push_back(m2); 

    std::vector<CSGNode> node_(3) ; 

    memcpy( node_.data(), node.data(), sizeof(CSGNode)*node_.size() );  

    CSGNode* n_ = node_.data(); 
    CSGNode::Dump( n_, node_.size(), "CSGNodeTest" );  
}

void test_zero()
{
    LOG(info); 

    CSGNode nd = {} ; 
    assert( nd.gtransformIdx() == 0u );  
    assert( nd.is_complement() == false );  

    unsigned tr = 42u ; 
    nd.setTransform( tr ); 

    assert( nd.gtransformIdx() == tr ); 
    assert( nd.is_complement() == false ); 

    nd.setComplement(true); 
    assert( nd.gtransformIdx() == tr ); 
    assert( nd.is_complement() == true ); 

    LOG(info) << nd.desc() ; 
}

void test_sphere()
{
    LOG(info); 
    CSGNode nd = CSGNode::Sphere(100.f); 
    LOG(info) << nd.desc() ; 
}

void test_change_transform()
{
    LOG(info); 

    CSGNode nd = {} ; 
    std::vector<unsigned> uu = {1001, 100, 10, 5, 6, 0, 20, 101, 206 } ; 
 
    // checking changing the transform whilst preserving the complement 

    for(int i=0 ; i < int(uu.size()-1) ; i++)
    {
        const unsigned& u0 = uu[i] ;  
        const unsigned& u1 = uu[i+1] ;  

        nd.setComplement( u0 % 2 == 0 ); 
        nd.setTransform(u0);   

        bool c0 = nd.is_complement(); 
        nd.zeroTransformComplement(); 

        nd.setComplement(c0) ; 
        nd.setTransform( u1 );   

        bool c1 = nd.is_complement(); 
        assert( c0 == c1 ); 

        unsigned u1_chk = nd.gtransformIdx();  
        assert( u1_chk == u1 ); 
    }
}

void test_Depth()
{
    //LOG(info); 
    for(unsigned i=0 ; i < 32 ; i++)
    {
        unsigned partIdxRel = i ; 
        unsigned depth = CSGNode::Depth(partIdxRel) ; 
        unsigned levelIdx = partIdxRel + 1 ; 

        std::cout 
            << " partIdxRel " << std::setw(4) << partIdxRel 
            << " partIdxRel+1 (dec) " << std::setw(4) << levelIdx  
            << " (bin) " <<  std::bitset<32>(levelIdx)
            << " depth " << std::setw(4) << depth
            << std::endl 
            ;
    }
}


void test_elv()
{
    CSGFoundry* fdl = CSGFoundry::Load(); 


    LOG(info) << "foundry " << fdl->desc() ; 
    fdl->summary(); 

    SBitSet* elv = SBitSet::Create( fdl->getNumMeshName(), "ELV", nullptr ) ;
    if(elv)
    {
        LOG(info) << elv->desc() << std::endl << fdl->descELV(elv) ;
    }
    CSGFoundry* fd = CSGCopy::Select(fdl, elv); 



    LOG(info) << fd->desc() ;     

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);  

    /*
    test_zero(); 
    test_sphere(); 
    test_copy();  
    test_change_transform();  
    test_Depth();  
    */

    test_elv(); 

    return 0 ; 
}
