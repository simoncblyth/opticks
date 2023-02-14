/**
stree_create_test.cc
=======================

Follow the structure of u4/U4Tree.h 

**/

#include <cstdlib>
#include "stree.h"
#include "stra.h"
#include "stran.h"

const char* FOLD = getenv("FOLD"); 

struct Dummy
{
    static std::string Dig(int nidx); 
    static int Boolean(int lvid); 
    static int LV(int nidx); 
    static int NumChild(int nidx); 
    static const Tran<double>* Transform(int nidx); 
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
inline int Dummy::Boolean(int lvid)
{
    int root = -1 ; 
    if( lvid == 4 )
    {
        int l = snd::Sphere(100.) ; 
        int r = snd::Box3(100.) ;         
        snd::SetNodeXForm(r,  stra<double>::Rotate( 0, 0, 1., 45., false ) ); 
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
inline const Tran<double>*  Dummy::Transform(int nidx)
{
    const Tran<double>* tr = nullptr ; 
    switch(nidx)
    {
        case -1: tr = Tran<double>::make_identity()                 ; break ; 
        case  0: tr = Tran<double>::make_identity()                 ; break ; 
        case  1: tr = Tran<double>::make_translate( 0., 0., 100.)   ; break ; 
        case  2: tr = Tran<double>::make_translate( 1000., 0., 0.)  ; break ; 
        case  3: tr = Tran<double>::make_rotate_a2b(   1., 0., 0., 0., 0., 1., false ) ; break ; 
        case  4: tr = Tran<double>::make_identity()                 ; break ; 
    }
    return tr ; 
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
        case  4:  root = Dummy::Boolean(lvid)             ; break ;   
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

void test_get_combined_transform( const stree& st )
{
    int lvid = 4 ; 

    std::vector<snode> nodes ; 
    st.find_lvid_nodes_(nodes, lvid); 

    std::vector<snd> nds ; 
    snd::GetLVID(nds, lvid);  

    int num_nodes = nodes.size() ; 
    int num_nds = nds.size(); 

    std::cout 
        << "test_get_combined_transform" << std::endl 
        << " lvid " << lvid << std::endl
        << " num_nodes (structural) : " << num_nodes 
        << std::endl
        << " num_nds (CSG) : " << num_nds 
        << std::endl
        ;  

    assert( num_nds == 3 ); 
    assert( num_nodes == 1 ); 

    const snode& node = nodes[0] ; 

    //for(int i=0 ; i < num_nds ; i++)
    int i = 1 ; 
    {
        const snd& nd = nds[i]; 
        Tran<double>* tv0 = Tran<double>::make_identity_(); 
        Tran<double>* tv1 = Tran<double>::make_identity_(); 

        st.get_combined_transform( tv0->t, tv0->v, node, &nd, nullptr  );   

        std::cout 
            << " i " << i 
            << std::endl 
            << st.desc_combined_transform( tv1->t, tv1->v, node, &nd )
            << std::endl
            << " tv0 " 
            << std::endl
            << tv0->desc() 
            << std::endl
            << " tv1 " 
            << std::endl
            << tv1->desc() 
            << std::endl
            ; 
    }
}

int main(int argc, char** argv)
{
    stree st ; 
    stree_create stc(&st);
    st.save(FOLD); 
    test_get_combined_transform(st); 
    return 0 ; 
}
 
