#include <cstdlib>
#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NNode.hpp"
#include "NTreeBuilder.hpp"


nnode* test_UnionTree_box(unsigned nprim, unsigned verbosity)
{
    std::vector<nnode*> prims ; 
    for(unsigned i=0 ; i < nprim ; i++)
    {
        nnode* a = new nbox(make_box3(400,400,400));
        a->verbosity = verbosity ; 
        prims.push_back(a);  
    }
    nnode* root = NTreeBuilder<nnode>::UnionTree(prims) ; 
    assert( root ) ; 
    root->verbosity = verbosity ; 
    return root ; 
}

nnode* test_UnionTree_box_Manual(unsigned nprim, unsigned verbosity)
{
    assert( nprim == 2);
    
    std::vector<nnode*> prims ; 
    for(unsigned i=0 ; i < nprim ; i++)
    {
        nnode* a = new nbox(make_box3(400,400,400));
        a->verbosity = verbosity ; 
        prims.push_back(a);  
    }
    nnode* root = new nunion( nunion::make_union( prims[0], prims[1]));
    assert( root ) ; 
    root->verbosity = verbosity ; 
    return root ; 
}

void dump(const nnode* n)
{
   LOG(info) << " dump /////////-------- " ; 

   nnode* l = n->left ; 
   nnode* r = n->right ; 
   assert( l ) ;  
   assert( r ) ;  

   std::cout << "N " << n->desc() << std::endl ; 
   std::cout << "L " << l->desc() << std::endl ; 
   std::cout << "R " << r->desc() << std::endl ; 
}


void test_UnionTree_bbox()
{
    unsigned nprim = 2 ;
    unsigned verbosity = 5 ;  
    nnode* ut0 = test_UnionTree_box_Manual(nprim, verbosity);
    dump(ut0);

    nbbox ut0_bb = ut0->bbox();
    LOG(info) << " ut0_bb " << ut0_bb.desc() ; 

    nnode* ut1 = test_UnionTree_box(nprim, verbosity);
    dump(ut1);

    nbbox ut1_bb = ut1->bbox();
    LOG(info) << " ut1_bb " << ut1_bb.desc() ; 
}



void test_UnionTree_sdf(unsigned nprim)
{
    LOG(info) << " nprim " << nprim  ; 

    unsigned verbosity = 5 ;  
    float x = 0 ; 
    float y = 0 ; 
    float z = 0 ; 

    nnode* ut0 = NULL ; 
    if(nprim == 2)
    {
        ut0 = test_UnionTree_box_Manual(nprim, verbosity);
        dump(ut0);
        std::function<float(float,float,float)> _sdf0 = ut0->sdf() ;
        float sd0 = _sdf0(x,y,z);
        LOG(info) << " sd0 " << sd0 ; 
    }

    nnode* ut1 = test_UnionTree_box(nprim, verbosity);
    dump(ut1);

    std::function<float(float,float,float)> _sdf1 = ut1->sdf() ;
    float sd1 = _sdf1(x,y,z);
    LOG(info) << " sd1 " << sd1 ; 
}

void test_bbox()
{
    nnode* a = new nbox(make_box3(400,400,400));
    a->verbosity = 5 ; 

    nbbox a_bb = a->bbox();
    LOG(info) 
        << " a " << a->desc()
        << " a_bb " << a_bb.desc()
        ;

    //nnode* b = new nnode(*a) ;       // <-- culprit : CAUSING THE INFINITE RECURSION
    //nnode* b = new nbox(*(nbox*)a) ;   // <--- fix
    nnode* b = a->make_copy();           // generalized fix   

    nbbox b_bb = b->bbox();
    LOG(info) 
        << " b " << b->desc()
        << " b_bb " << b_bb.desc()
        ;
}

void test_bbox_2()
{
    nnode* o = new nbox(make_box3(200,200,200));
    o->verbosity = 5 ; 
    nnode* a = new nbox(make_box3(400,400,400));
    a->verbosity = 5 ; 

    std::vector<nnode*> prims ; 
    prims.push_back(o); 
    prims.push_back(a); 


    nbbox a_bb = a->bbox();
    LOG(info) 
        << " a " << a->desc()
        << " a_bb " << a_bb.desc()
        ;

    std::vector<nnode*> cprims ; 
    cprims = prims ; 

    std::reverse(cprims.begin(), cprims.end()); 

    nnode* a2 = cprims.back() ; 
    nnode* b = a2->make_copy();
    cprims.pop_back(); 

 
    nbbox b_bb = b->bbox();
    LOG(info) 
        << " b " << b->desc()
        << " b_bb " << b_bb.desc()
        ;

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned nprim = argc > 1 ? atoi(argv[1]) : 2  ; 

    //test_UnionTree_bbox() ; 
    //test_bbox() ; 
    //test_bbox_2() ; 

    test_UnionTree_sdf(nprim) ; 

    return 0 ; 
}

/*

2018-06-13 19:37:56.887 INFO  [19092658] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
NNodeAnalyse height 3 count 15
                              un                            

              un                              un            

      un              un              un              un    

  ze      ze      ze      ze      ze      ze      ze      ze


2018-06-13 19:37:56.888 INFO  [19092658] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
NNodeAnalyse height 3 count 15
                              un                            

              un                              un            

      un              un              un              un    

  bo      bo      bo      bo      bo      ze      ze      ze


2018-06-13 19:37:56.888 INFO  [19092658] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
NNodeAnalyse height 3 count 9
                              un    

              un                  bo

      un              un            

  bo      bo      bo      bo        


2018-06-13 19:37:56.888 INFO  [19092658] [*NTreeBuilder<nnode>::CommonTree@19]  num_prims 5 height 3 operator union

*/
