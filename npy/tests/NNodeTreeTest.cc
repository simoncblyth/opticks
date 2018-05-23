
#include "NPrimitives.hpp"
#include "NGLMExt.hpp"
#include "NNode.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


void test_collect_ancestors(const nnode& prim, unsigned expect)
{
    std::vector<const nnode*> ancestors ;
    prim.collect_ancestors(ancestors, CSG_ZERO);

    LOG(info) << "test_collect_ancestors " << ancestors.size() << " expect " << expect ; 
    assert(ancestors.size() == expect );
}
void test_collect_connectedtype_ancestors(const nnode& prim, unsigned expect)
{
    std::vector<const nnode*> ancestors ;
    prim.collect_connectedtype_ancestors(ancestors);

    LOG(info) << "test_collect_connectedtype_ancestors " << ancestors.size() << " expect " << expect ; 
    assert(ancestors.size() == expect );
}
void test_collect_monogroup(const nnode& prim, unsigned expect)
{
    std::vector<const nnode*> uniongroup ;
    prim.collect_monogroup(uniongroup);

    LOG(info) << "test_collect_monogroup " << uniongroup.size() << " expect " << expect ; 
    assert(uniongroup.size() == expect );
}
void test_is_same_union(const nnode& iprim, const nnode& jprim, bool expect)
{
    bool same = nnode::is_same_union(&iprim, &jprim);
    LOG(info) << "test_is_same_union " << same << " expect " << expect ; 
}


void test_uncycy()
{
    LOG(info) << "test_uncycy" ;

    /*
              un(ab)
           
           cy(a)   cy(b)   

    */

    ncylinder a = make_cylinder(0.000,0.000,0.000,650.000,-23.500,23.500,0.000,0.000) ; a.label = "a" ;   
    ncylinder b = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   
    b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,0.000,-41.000,1.000) ;
    nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   

    test_collect_ancestors(a, 1);
    test_collect_connectedtype_ancestors(a, 1);
    test_collect_monogroup(a, 2);
    test_is_same_union(a,b,true);
}


void test_uncycycy()
{
    LOG(info) << "test_uncycycy" ;

    /*

                         un(abc)
              un(ab)             cy(c)
           
           cy(a)   cy(b)   
    */


    ncylinder a = make_cylinder(0.000,0.000,0.000,650.000,-23.500,23.500,0.000,0.000) ; a.label = "a" ;   
    ncylinder b = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   
    b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,0.000,-41.000,1.000) ;
    nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   

    ncylinder c = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   
    nunion abc = make_union(&ab, &c) ; abc.label = "abc" ; ab.parent = &abc ; c.parent = &abc ;  ;   

    test_collect_ancestors(a, 2);
    test_collect_connectedtype_ancestors(a, 2);

    test_collect_monogroup(a, 3);
    test_is_same_union(a,b,true);
    test_is_same_union(a,c,true);
    test_is_same_union(b,c,true);
}


void test_di_uncycycy_co()
{
    LOG(info) << "test_di_uncycycy_co" ;

    /*

                                           di(abcd)

                         un(abc)                     co(d)
              un(ab)             cy(c)
           
           cy(a)   cy(b)   
    */


    ncylinder a = make_cylinder(0.000,0.000,0.000,650.000,-23.500,23.500,0.000,0.000) ; a.label = "a" ;   
    ncylinder b = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   
    b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,0.000,-41.000,1.000) ;
    nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   

    ncylinder c = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   
    nunion abc = make_union(&ab, &c) ; abc.label = "abc" ; ab.parent = &abc ; c.parent = &abc ;  ;   

    ncone d = make_cone(650.,23.5,750.,33.5) ; d.label = "d" ;     // NB param dont make much sense, just testing tree
    ndifference abcd = make_difference(&abc, &d) ; abcd.label = "abcd" ; abc.parent = &abcd ; d.parent = &abcd ;  ;   


    test_collect_ancestors(a, 3);
    test_collect_connectedtype_ancestors(a, 2);

    test_collect_monogroup(a, 3);
    test_is_same_union(a,b,true);
    test_is_same_union(a,c,true);
    test_is_same_union(b,c,true);
}


void test_di_uncycycy_uncycy()
{
    LOG(info) << "test_di_uncycycy_uncycy" ;

    /*

                                           di(abcde)

                         un(abc)                     un(de)
              un(ab)             cy(c)           cy(d)     cy(e)
           
           cy(a)   cy(b)   
    */


    ncylinder a = make_cylinder(0.000,0.000,0.000,650.000,-23.500,23.500,0.000,0.000) ; a.label = "a" ;   
    ncylinder b = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   

    b.transform = nmat4triple::make_transform(1.000,0.000,0.000,0.000,  0.000,1.000,0.000,0.000,  0.000,0.000,1.000,0.000,  0.000,0.000,-41.000,1.000) ;
    nunion ab = make_union(&a, &b) ; ab.label = "ab" ; a.parent = &ab ; b.parent = &ab ;  ;   

    ncylinder c = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; b.label = "b" ;   
    nunion abc = make_union(&ab, &c) ; abc.label = "abc" ; ab.parent = &abc ; c.parent = &abc ;  ;   



    ncylinder d = make_cylinder(0.000,0.000,0.000,650.000,-23.500,23.500,0.000,0.000) ; d.label = "d" ;   
    ncylinder e = make_cylinder(0.000,0.000,0.000,31.500,-17.500,17.500,0.000,0.000) ; e.label = "e" ;   
    nunion de = make_union(&d, &e) ; de.label = "de" ; d.parent = &de ; e.parent = &de ;  ;   

    ndifference abcde = make_difference(&abc, &de) ; abcde.label = "abcde" ; abc.parent = &abcde ; de.parent = &abcde ;  ;   


    test_collect_ancestors(a, 3);
    test_collect_connectedtype_ancestors(a, 2);

    test_collect_monogroup(a, 3);
    test_is_same_union(a,b,true);
    test_is_same_union(a,c,true);
    test_is_same_union(b,c,true);


    test_is_same_union(a,d,false);
    test_is_same_union(a,e,false);
    test_is_same_union(d,e,true);

}






int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_uncycy();
    test_uncycycy();
    test_di_uncycycy_co();
    test_di_uncycycy_uncycy();

    return 0 ; 
}



