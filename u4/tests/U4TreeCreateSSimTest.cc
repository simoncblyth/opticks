/**
U4TreeCreateSSimTest.cc
=========================

::

    ~/o/u4/tests/U4TreeCreateSSimTest.sh 

    TEST=find_lvid ~/o/u4/tests/U4TreeCreateSSimTest.sh 
    TEST=pick_lvid_ordinal_node ~/o/u4/tests/U4TreeCreateSSimTest.sh
    TEST=get_combined_transform  ~/o/u4/tests/U4TreeCreateSSimTest.sh
    TEST=get_combined_tran_and_aabb ~/o/u4/tests/U4TreeCreateSSimTest.sh


1. access geometry with U4VolumeMaker::PV
2. create SSim instance with empty stree and SScene
3. populate the stree and SScene instances with U4Tree::Create and SSim::initSceneFromTree
4. save the SSim to $BASE directory which appends reldir "SSim"
5. run method selected by TEST envvar

See U4TreeCreateSSimLoadTest.cc for loading the saved SSim 

**/

#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "stra.h"
#include "SSim.hh"
#include "U4VolumeMaker.hh"
#include "U4Tree.h"


struct U4TreeCreateSSimTest
{
    static constexpr const char* _LEVEL = "U4TreeCreateSSimTest__LEVEL" ; 
    int level ;  
    SSim*   sim ; 
    stree*  st ; 
    U4Tree* tr ; 

    bool starting = true ; 
    const char* q_soname ; 
    int q_lvid ; 
    int q_lvid_ordinal = 0 ; // first one 
    const snode* q_node ; 
    const sn* q_nd ; 
    int q_nd_num_child ;

    U4TreeCreateSSimTest( const G4VPhysicalVolume* world ); 
    void init(); 
    std::string desc(const char* label) const ;

    int find_lvid() const; 
    int pick_lvid_ordinal_node() const ; 
    int get_combined_transform() const; 
    int get_combined_tran_and_aabb() const; 
    int GetLVListnodes() const ; 

    int run_TEST_method() const ; 
    static int Main(int argc, char** argv);

};

U4TreeCreateSSimTest::U4TreeCreateSSimTest(const G4VPhysicalVolume* world )
    :
    level(ssys::getenvint(_LEVEL,2)),
    sim(SSim::Create()),
    st(sim->tree),
    tr(world ? U4Tree::Create(st, world) : nullptr),
    q_soname(ssys::getenvvar("GEOM", nullptr)),
    q_lvid( q_soname ? st->find_lvid(q_soname, starting) : -1 ), 
    q_node( q_lvid > -1 ? st->pick_lvid_ordinal_node(q_lvid, q_lvid_ordinal ) : nullptr ),
    q_nd(q_lvid > -1 ? sn::GetLVRoot(q_lvid) : nullptr ),  
    q_nd_num_child( q_nd ? q_nd->num_child() : -1) 
{
    init();
}

void U4TreeCreateSSimTest::init()
{
    assert(tr); 
    sim->initSceneFromTree();  
    std::cerr << " save SSim to $FOLD " << std::endl ; 
    sim->save("$FOLD");  // "SSim" reldir added by the save  
    LOG(info) << " sim.tree.desc " << std::endl << st->desc() ;
}


std::string U4TreeCreateSSimTest::desc(const char* label) const 
{
    std::stringstream ss ; 
    ss << label << "\n"
       << " q_soname " << ( q_soname ? q_soname : "-" )
       << " q_lvid " << q_lvid 
       << " q_lvid_ordinal " << q_lvid_ordinal 
       << " q_node.desc\n" 
       << ( q_node ? q_node->desc() : "-" )
       << "\n"
       << " q_nd.desc\n"
       << ( q_nd ? q_nd->desc() : "-" ) 
       << "\n"
       << " q_nd_num_child " << q_nd_num_child
       << "\n"
       ;

    std::string str = ss.str() ; 
    return str ; 
}


int U4TreeCreateSSimTest::find_lvid() const 
{
    int rc = q_lvid > -1 ? 0 : 1 ; 
    std::cout 
        << desc("U4TreeCreateSSimTest::find_lvid")
        << " rc " << rc 
        << "\n" 
        ; 
    return rc ; 
}

int U4TreeCreateSSimTest::pick_lvid_ordinal_node() const
{
    int rc = q_node == nullptr ? 1 : 0 ; 
    std::cout 
        << desc("U4TreeCreateSSimTest::pick_lvid_ordinal_node")
        << " rc " << rc 
        << "\n" 
        ; 
    return rc ; 
}



/**
U4TreeCreateSSimTest::get_combined_transform
----------------------------------------------

Checking combined transforms for all child nodes
of the selected root lvid node "q_nd"

**/


int U4TreeCreateSSimTest::get_combined_transform() const
{
    if(!q_node) return 1 ; 
    if(!q_nd) return 1 ; 

    std::cout << desc("U4TreeCreateSSimTest::get_combined_transform") ; 

    for(int i=0 ; i < 3 ; i++)
    { 
        for(int j=0 ; j < q_nd_num_child ; j++)
        {
            sn* c = q_nd->get_child(j);
            if( i == 0 ) std::cout << " sn::get_child/desc      " << std::setw(3) << j << " : " << c->desc() << "\n" ;  
            if( i == 1 ) std::cout << " sn::get_child/desc_prim " << std::setw(3) << j << " : " << c->desc_prim() << "\n" ;  
            if( i == 2 ) std::cout << " sn::get_child/descXF    " << std::setw(3) << j << " : " << c->descXF() << "\n" ;  
        }
        std::cout << "\n" ; 
    }
 
    for(int j=-1 ; j < q_nd_num_child ; j++)
    {
        const sn* n = j == -1 ? q_nd : q_nd->get_child(j) ;

        glm::tmat4x4<double> t(1.); 
        glm::tmat4x4<double> v(1.); 
        std::stringstream ss ; 
        std::ostream* out = level > 2 ? &ss : nullptr ; 

        st->get_combined_transform(t,v,*q_node,n, out) ;  
        std::string str = out ? ss.str() : "-" ; 

        std::cout
            << " q_nd+child " << std::setw(3) << j << " : " << n->desc() 
            << "\n"
            <<  str
            << "\n"
            << stra<double>::Desc(t,v, "t", "v") 
            << "\n"
            ;
   } 

   return 0 ; 
}


/**
U4TreeCreateSSimTest::get_combined_tran_and_aabb
-------------------------------------------------

Note similarity with CSGImport::importNode

**/


int U4TreeCreateSSimTest::get_combined_tran_and_aabb() const
{
    if(!q_node) return 1 ; 
    if(!q_nd) return 1 ; 

    std::cout << desc("U4TreeCreateSSimTest::get_combined_tran_and_aabb") ; 


    for(int j=-1 ; j < q_nd_num_child ; j++)
    {
        const sn* nd = j == -1 ? q_nd : q_nd->get_child(j) ;

        int typecode = nd ? nd->typecode : CSG_ZERO ; 
        bool leaf = CSG::IsLeaf(typecode) ;

        std::stringstream ss ; 
        std::ostream* out = level > 2 ? &ss : nullptr ; 

        std::array<double,6> bb ; 
        double* aabb = leaf ? bb.data() : nullptr ;

        const Tran<double>* tran = leaf ? st->get_combined_tran_and_aabb(aabb,*q_node, nd, out) : nullptr ;  

        std::string str = out ? ss.str() : "-" ; 
        std::cout
            << " q_nd+child " << std::setw(3) << j << " : " << nd->desc() 
            << "\n"
            <<  str
            << "\n"
            << ( tran ? stra<double>::Desc(tran->t,tran->v, "t", "v") : "no-tran-for-non-leaf" )  
            << "\n"
            << ( aabb ? s_bb::Desc(aabb) : "no-aabb-for-non-leaf" )
            << "\n"
            ;
   } 


   return 0 ;  
}

int U4TreeCreateSSimTest::GetLVListnodes() const 
{
    std::vector<const sn*> lns ; 
    sn::GetLVListnodes( lns, q_lvid ); 
    int num_lns = lns.size(); 

    int child_total = sn::GetChildTotal(lns) ; 


    std::cout 
        << desc("U4TreeCreateSSimTest::GetLVListnodes") 
        << " num_lns  " << num_lns 
        << " child_total " << child_total 
        << "\n"
        ; 

    for(int i=0 ; i < num_lns ; i++)
    {
        const sn* n = lns[i]; 
        std::cout 
            << std::setw(3) << i << " : " << n->desc() << "\n" 
            ;  
    }

    return 0 ; 
}


int U4TreeCreateSSimTest::run_TEST_method() const
{
    const char* TEST = ssys::getenvvar("TEST", "NONE") ; 
    int rc = 0 ; 
    if(     strcmp(TEST, "find_lvid") == 0 )                  rc = find_lvid();  
    else if(strcmp(TEST, "pick_lvid_ordinal_node") == 0 )     rc = pick_lvid_ordinal_node();  
    else if(strcmp(TEST, "get_combined_transform") == 0 )     rc = get_combined_transform();  
    else if(strcmp(TEST, "get_combined_tran_and_aabb") == 0 ) rc = get_combined_tran_and_aabb();  
    else if(strcmp(TEST, "GetLVListnodes") == 0 )             rc = GetLVListnodes();  
    else std::cout << "U4TreeCreateSSimTest::run_TEST_method no-impl for TEST[" << ( TEST ? TEST : "-" ) << "]" ;  
    return rc ; 
}

int U4TreeCreateSSimTest::Main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
    LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;   
    if(world == nullptr) return 0 ; 

    U4TreeCreateSSimTest test(world); 
    return test.run_TEST_method() ; 
}


int main(int argc, char** argv)
{
    return U4TreeCreateSSimTest::Main(argc, argv); 
}
