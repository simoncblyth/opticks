/**
stree_create_test.cc
=======================

This test creates a simple geometry from scratch, following 
the general approach of u4/U4Tree.h, and then uses that geometry to 
compare query results with expections, for example checking::

    stree::get_combined_transform 

For querying with more complex loaded geometries see *stree_load_test.cc**
**/

#include <cstdlib>
#include "stree.h"
#include "stra.h"
#include "stran.h"
#include "ssys.h"

const char* FOLD = getenv("FOLD"); 
const bool VERBOSE = ssys::getenvbool("VERBOSE"); 
const int LVID = ssys::getenvint("LVID", 4); 

struct Dummy
{
    static std::string Dig(int nidx); 
    static int Boolean(int lvid, int it); 
    static int LV(int nidx); 
    static int NumChild(int nidx); 
    static const Tran<double>* Transform(int it); 
    static const Tran<double>* Product( const std::vector<int>& its); 
    static const Tran<double>* Expected(int ie); 
    static int Solid(int lvid); 
    static const char* SolidName(int lvid); 
    static constexpr const int NUM_LVID = 5 ; 
};

inline std::string Dummy::Dig(int nidx)
{
    std::stringstream ss ; 
    ss << "Dummy" << nidx ; 
    std::string str = ss.str(); 
    return str ; 
}
inline int Dummy::Boolean(int lvid, int it)
{
    int root = -1 ; 
    if( lvid == 4 )
    {
        int l = snd::Sphere(100.) ; 
        int r = snd::Box3(100.) ;         
        const Tran<double>* tv = Transform(it) ;  
        snd::SetNodeXForm(r, tv->t, tv->v );  
        root = snd::Boolean(CSG_UNION, l, r);   
    }
    return root ; 
}
inline int Dummy::LV(int nidx)
{
    int lv = 0 ; 
    switch(nidx)
    {
        case 0: lv = 0 ; break ; 
        case 1: lv = 1 ; break ; 
        case 2: lv = 2 ; break ; 
        case 3: lv = 3 ; break ; 
        case 4: lv = 4 ; break ; 
    }
    return lv ; 
}
inline int Dummy::NumChild(int nidx)
{
    int nc = 0 ; 
    switch(nidx)
    {
        case 0: nc = 1 ; break ; 
        case 1: nc = 1 ; break ; 
        case 2: nc = 1 ; break ; 
        case 3: nc = 1 ; break ; 
        case 4: nc = 0 ; break ; 
    }
    return nc ; 
}
inline const Tran<double>*  Dummy::Transform(int it)
{
    const Tran<double>* tr = nullptr ; 
    switch(it)
    {
        case -1: tr = Tran<double>::make_identity()                 ; break ; 
        case  0: tr = Tran<double>::make_identity()                 ; break ; 
        case  1: tr = Tran<double>::make_translate( 0., 0., 100.)   ; break ; 
        case  2: tr = Tran<double>::make_translate( 1000., 0., 0.)  ; break ; 
        case  3: tr = Tran<double>::make_rotate_a2b(   1., 0., 0., 0., 0., 1., false ) ; break ; 
        case  4: tr = Tran<double>::make_identity()                 ; break ; 
        case 100: tr = Tran<double>::make_rotate( 0., 0., 1., 45. ) ; break ; 
    }
    assert(tr); 
    return tr ; 
}

inline const Tran<double>* Dummy::Product(const std::vector<int>& its)
{
    std::vector<const Tran<double>*> trs ; 
    int num_tr = its.size(); 
    for(int i=0 ; i < num_tr ; i++)
    {
        int it = its[i] ; 
        const Tran<double>* tr = Transform(it); 
        trs.push_back(tr); 
    }
    const Tran<double>* prd = Tran<double>::product( trs, false );  
    return prd ; 
}

inline const Tran<double>* Dummy::Expected(int ie)
{
    std::vector<int> its = {0,1,2,3,4} ; 
    if( ie == 1 ) its.push_back(100) ; 
    return Product(its) ; 
}
inline int Dummy::Solid(int lvid)
{
    int root = -1 ; 
    switch(lvid)
    {
        case  0:  root = snd::Sphere(100.)                ; break ; 
        case  1:  root = snd::Box3(100.)                  ; break ; 
        case  2:  root = snd::Cylinder(100., -10., 10. )  ; break ; 
        case  3:  root = snd::Box3(100.);                 ; break ; 
        case  4:  root = Dummy::Boolean(lvid, 100)        ; break ;   
    }
    return root ; 
}
inline const char* Dummy::SolidName(int lvid)
{
    const char* n = nullptr ; 
    switch(lvid)
    {
        case 0: n = "0_Sphere"     ; break ; 
        case 1: n = "1_Box"        ; break ; 
        case 2: n = "2_Cylinder"   ; break ; 
        case 3: n = "3_Box"        ; break ; 
        case 4: n = "4_Boolean"    ; break ; 
    }
    return strdup(n); 
}


struct stree_create
{
    stree* st ; 
    stree_create( stree* st_) ; 
    void init(); 
    void initSolids(); 
    void initSolid(int lvid); 
    void initNodes(); 
    int  initNodes_r( int depth, int sibdex, int parent  ); 
}; 

inline stree_create::stree_create( stree* st_)
    :
    st(st_)
{
    init(); 
}
inline void stree_create::init()
{   
    initSolids(); 
    initNodes(); 
}
inline void stree_create::initSolids()
{
    for(int lvid=0 ; lvid < Dummy::NUM_LVID ; lvid++) initSolid(lvid); 
}
inline void stree_create::initSolid(int lvid )
{
    assert( int(st->solids.size()) == lvid );

    int root = Dummy::Solid(lvid);
    assert( root > -1 );
    snd::SetLVID(root, lvid );

    const char* name = Dummy::SolidName(lvid) ; 

    st->soname.push_back(name);
    st->solids.push_back(root);
}
inline void stree_create::initNodes()
{   
    int nidx = initNodes_r(0, -1, -1 );
    assert( 0 == nidx );
}
inline int stree_create::initNodes_r( int depth, int sibdex, int parent  )
{
    int nidx = st->nds.size() ;  // 0-based node index

    int copyno = nidx ; 
    int boundary = 0 ; 

    int lvid = Dummy::LV(nidx); 
    std::string dig = Dummy::Dig(nidx) ;  
    int num_child = Dummy::NumChild(nidx) ; 

    const Tran<double>* tv = Dummy::Transform(nidx); 
    //const Tran<double>* tv = Dummy::Transform(-1);    // -1 switches to all identity transforms

    snode nd ; 

    nd.index = nidx ;
    nd.depth = depth ;   
    nd.sibdex = sibdex ; 
    nd.parent = parent ;   

    nd.num_child = num_child ; 
    nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 

    nd.copyno = copyno ; 
    nd.sensor_id = -1 ;  
    nd.sensor_index = -1 ;
    nd.repeat_index = 0 ; 

    nd.repeat_ordinal = -1 ;
    nd.boundary = boundary ; 

    st->nds.push_back(nd); 
    st->digs.push_back(dig); 
    st->m2w.push_back(tv->t);  
    st->w2m.push_back(tv->v);  

    if(sibdex == 0 && nd.parent > -1) st->nds[nd.parent].first_child = nd.index ;
    // record first_child nidx into parent snode by reaching up thru the recursion levels 

    int p_sib = -1 ;
    int i_sib = -1 ;
    for (int i=0 ; i < num_child ; i++ )
    {
        p_sib = i_sib ;    // node index of previous child gets set for i > 0
        i_sib = initNodes_r( depth+1, i, nd.index );

        if(p_sib > -1) st->nds[p_sib].next_sibling = i_sib ;
        // after first child : reach back to previous sibling snode to set the sib->sib linkage, default -1
    }
    return nd.index ;
}

void test_get_combined_transform( const stree& st, int lvid )
{

    std::vector<snode> nodes ; 
    st.find_lvid_nodes_(nodes, lvid); 

    std::vector<snd> nds ; 
    snd::GetLVID(nds, lvid);  

    int num_nodes = nodes.size() ; 
    int num_nds = nds.size(); 

    std::cout 
        << "test_get_combined_transform" << std::endl 
        << " lvid " << lvid << std::endl
        << " nodes(vols) : " << num_nodes 
        << std::endl
        << " nds(csg) : " << num_nds 
        << std::endl
        ;  

    std::cout << "nodes(vols)" << std::endl << snode::Brief_(nodes) ; 
    std::cout << "nds(csg)" << std::endl << snd::Brief_(nds) ; 

    assert( num_nodes == 1 ); 
    assert( num_nds == 3 ); 

    const snode& node = nodes[0] ; 


    std::stringstream ss ; 
    for(int i=0 ; i < num_nds ; i++)
    {
        const snd& nd = nds[i]; 
        Tran<double>* tv = Tran<double>::make_identity_(); 

        const Tran<double>* ex = Dummy::Expected(i); 

        ss << " i " << i 
           << std::endl 
           ; 

        std::ostream* out = VERBOSE ? &ss : nullptr ; 
        st.get_combined_transform(tv->t, tv->v, node, &nd, out );

        ss  << " tv " 
            << std::endl
            << tv->desc() 
            << std::endl
            << " expected " 
            << std::endl
            << ex->desc() 
            << std::endl
            ; 
    }

   std::string str = ss.str(); 
   std::cout << str ; 

}

int main(int argc, char** argv)
{
    stree st ; 
    stree_create stc(&st);
    st.save(FOLD); 
    test_get_combined_transform(st, LVID); 
    return 0 ; 
}
 
