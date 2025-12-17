/**

~/o/u4/tests/U4SolidMakerTest.sh

**/


#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "G4VSolid.hh"
#include "U4Mesh.h"
#include "U4Solid.h"
#include "U4SolidMaker.hh"

#include "s_csg.h"
#include "sn.h"


struct U4SolidMakerTest
{
    static const char* SOLID ;
    static const char* TEST ;
    static const int   LEVEL ;

    static const G4VSolid* MakeSolid();
    static sn* Convert_(const G4VSolid* solid);

    static void get_node_bb(int lvid);
    static int Convert();
    static int Main(int argc, char** argv);
};

const char* U4SolidMakerTest::SOLID = ssys::getenvvar("SOLID", "WaterDistributer");
const char* U4SolidMakerTest::TEST = ssys::getenvvar("TEST", "Convert");
const int  U4SolidMakerTest::LEVEL = ssys::getenvint("LEVEL", 4 );

inline int U4SolidMakerTest::Main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Convert")) rc += Convert();
    return rc ;
}

inline const G4VSolid* U4SolidMakerTest::MakeSolid()
{
    bool can_make = U4SolidMaker::CanMake(SOLID);
    if(!can_make)  std::cerr << "U4SolidMakerTest::MakeSolid SOLID[" << ( SOLID ? SOLID : "-" ) << "] can_make[" << ( can_make ? "Y" : "N" ) << "]\n" ;
    if(!can_make) return 0 ;
    const G4VSolid* solid = U4SolidMaker::Make(SOLID);
    return solid ;
}

inline sn* U4SolidMakerTest::Convert_(const G4VSolid* solid )
{
    int lvid = 0 ;
    int depth = 0 ;
    sn* nd = U4Solid::Convert(solid, lvid, depth, LEVEL);
    return nd ;
}


/**
U4SolidMakerTest::get_node_bb
------------------------------

See stree::get_node_bb

**/


inline void U4SolidMakerTest::get_node_bb(int lvid)
{
    typedef std::array<double,6> BB ;
    typedef std::array<double,4> CE ;
    BB bb = {} ;

    typedef std::vector<BB> VBB ;



    std::vector<const sn*> bds ;
    sn::GetLVNodesComplete(bds, lvid);
    int bn = bds.size();

    std::vector<const sn*> lns ;
    sn::GetLVListnodes( lns, lvid );
    int ln = lns.size();

    std::vector<const sn*> subs ;

    for(int i=0 ; i < bn ; i++)
    {
        const sn* n = bds[i];
        //int  typecode = n ? n->typecode : CSG_ZERO ;

        if(n && n->is_listnode())
        {
            int num_sub = n->child.size() ;
            for(int j=0 ; j < num_sub ; j++)
            {
                const sn* c = n->child[j];
                subs.push_back(c);
            }
        }
    }

    int ns = subs.size();


    std::cout << "[U4SolidMakerTest::get_node_bb lvid " << lvid << "\n" ;
    std::cout << " bn " << bn << "\n" ;
    std::cout << " ln " << ln << "\n" ;
    std::cout << " ns " << ns << "\n" ;

    std::ostream* out = nullptr ;

    VBB vbb0 = {} ;
    VBB vbb  = {} ;
    VBB vbb1 = {} ;

    for( int i=0 ; i < ns ; i++ )
    {
        const sn* n = subs[i];
        bool leaf = CSG::IsLeaf(n->typecode) ;
        assert(leaf);
        if(!leaf) continue ;

        BB n_bb0 = {} ;
        n->copyBB_data(n_bb0.data()) ; // without transform
        vbb0.push_back(n_bb0);

        BB n_bb = {} ;
        n->copyBB_data( n_bb.data() );

        glm::tmat4x4<double> tc(1.) ;
        glm::tmat4x4<double> vc(1.) ;
        sn::NodeTransformProduct(n->idx(), tc, vc, false, out, nullptr );  // reverse:false

        stra<double>::Transform_AABB_Inplace(n_bb.data(), tc);
        vbb.push_back(n_bb);

        s_bb::IncludeAABB( bb.data(), n_bb.data(), out );
        vbb1.push_back(bb);
    }

    assert( vbb0.size() == vbb.size() );
    assert( vbb0.size() == vbb1.size() );

    size_t num = vbb0.size();
    for(size_t i=0 ; i < num ; i++ )
    {
        std::cout
           << std::setw(2) << i
           << " "
           << s_bb::Desc(vbb0[i].data())
           << " "
           << s_bb::Desc(vbb[i].data())
           << " "
           << s_bb::Desc(vbb1[i].data())
           << " "
           << "\n"
           ;
    }



    CE ce = {};
    s_bb::CenterExtent( ce.data(), bb.data() );

    std::cout << " bb " << s_bb::Desc(bb.data()) << "\n" ;
    std::cout << " ce " << s_bb::Desc_<double,4>(ce.data()) << "\n" ;


    std::cout << "]U4SolidMakerTest::get_node_bb lvid " << lvid << "\n" ;
}


inline int U4SolidMakerTest::Convert()
{
    const G4VSolid* solid = MakeSolid();
    NPFold* fold = U4Mesh::Serialize(solid) ;
    fold->set_meta<std::string>("SOLID",SOLID);
    fold->set_meta<std::string>("desc","placeholder-desc");
    fold->save("$FOLD", SOLID );

    s_csg* _csg = new s_csg ;  // hold pools (normally stree holds _csg)
    assert( _csg );

    sn* nd = Convert_(solid);

    std::cout
        << "[U4SolidMakerTest nd.desc\n"
        <<   nd->desc()
        << "]U4SolidMakeTest nd.desc\n"
        ;

    if(LEVEL > 2 ) std::cout
        << "\n[U4SolidMakerTest nd->render() \n"
        << nd->render()
        << "\n]U4SolidMakerTest nd->render() \n\n"
        ;

    if(LEVEL > 3 ) std::cout
        << "\n[U4SolidMakerTest nd->detail_r()\n"
        << nd->detail_r()
        << "\n]U4SolidMakerTest nd->detail_r() \n\n"
        ;

    if(LEVEL > 3 ) std::cout
        << "\n[U4SolidMakerTest  nd->desc_prim_all() \n"
        << nd->desc_prim_all(false)
        << "\n]U4SolidMakerTest  nd->desc_prim_all() \n"
        ;


    int lvid = 0 ;
    get_node_bb(lvid);



    std::cout << "sn::Desc.0.before-delete-expect-some-nodes\n"  << sn::Desc() << "\n" ;
    delete nd ;
    std::cout << "sn::Desc.1.after-delete-expect-no-nodes\n"  << sn::Desc() << "\n" ;

    return 0;
}


int main(int argc, char** argv)
{
    return U4SolidMakerTest::Main(argc, argv);
}
