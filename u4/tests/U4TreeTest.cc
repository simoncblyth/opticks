#include "OPTICKS_LOG.hh"


#include "snd.h"
std::vector<snd> snd::node  = {} ; 
std::vector<spa> snd::param = {} ; 
std::vector<sxf> snd::xform = {} ; 
std::vector<sbb> snd::aabb  = {} ; 
// HMM: how to avoid ? 




#include "SPath.hh"
#include "SSys.hh"
#include "U4GDML.h"
#include "U4Tree.h"


const char* FOLD = SPath::Resolve("$TMP/U4TreeTest", DIRPATH); 


void test_saveload_get_children(const stree* tree0, const stree* tree1, int nidx )
{
    std::vector<int> children0 ; 
    std::vector<int> children1 ; 
    tree0->get_children(children0, nidx); 
    tree1->get_children(children1, nidx); 

    if( nidx % 10000 == 0 )
    std::cout << " nidx " << nidx << " children " << stree::Desc(children0) << std::endl ; 

    assert( stree::Compare(children0, children1) == 0 ); 
}
void test_saveload_get_children(const stree* tree0, const stree* tree1)
{
    std::cout << "[ test_saveload_get_children " << std::endl ; 
    assert( tree0->nds.size() == tree1->nds.size() ); 
    for(int nidx=0 ; nidx < int(tree0->nds.size()) ; nidx++)
        test_saveload_get_children(tree0, tree1, nidx ) ; 
    std::cout << "] test_saveload_get_children " << std::endl ; 
}

void test_saveload_get_progeny(const stree* tree0, const stree* tree1)
{
    std::cout << "test_saveload_get_progeny_r " << std::endl ; 
    int nidx = 0 ; 
    std::vector<int> progeny0 ; 
    std::vector<int> progeny1 ; 
    tree0->get_progeny(progeny0, nidx); 
    tree1->get_progeny(progeny1, nidx); 
    assert( stree::Compare(progeny0, progeny1) == 0 ); 
    std::cout << " nidx " << nidx << " progeny " << stree::Desc(progeny0) << std::endl ; 
}

void test_saveload(const stree* st0)
{
    std::cout << "[ st0.save " << std::endl ; 
    st0->save(FOLD); 
    std::cout << "] st0.save  " << st0->desc() << std::endl ; 

    stree st1 ; 
    std::cout << "[ st1.load " << std::endl ; 
    st1.load(FOLD);  
    std::cout << "] st1.load " << st1.desc() << std::endl ; 

    test_saveload_get_children(st0, &st1); 
    test_saveload_get_progeny( st0, &st1); 
}

void test_load(const char* fold)
{
    stree st ;
    st.load(fold); 
    std::cout << "st.desc_sub" << std::endl << st.desc_sub() << std::endl ; 

    // see sysrap/tests/stree_test.cc for stree exercises 
}


void test_get_pv_0(const U4Tree* tree)
{
    std::cout << "[ test_get_pv_0 " << std::endl ; 
    const stree* st = tree->st ; 

    const char* q_spec0 = SSys::getenvvar("SPEC0", "HamamatsuR12860sMask_virtual:0:0" ); 
    const char* q_spec1 = SSys::getenvvar("SPEC1", "HamamatsuR12860sMask_virtual:0:-1" );

    int nidx0 = st->find_lvid_node(q_spec0 );   
    int nidx1 = st->find_lvid_node(q_spec1 );   

    const G4VPhysicalVolume* const pv0_ = tree->get_pv(nidx0) ;  
    const G4VPhysicalVolume* const pv1_ = tree->get_pv(nidx1) ;  

    const G4PVPlacement* pv0 = dynamic_cast<const G4PVPlacement*>(pv0_) ; 
    const G4PVPlacement* pv1 = dynamic_cast<const G4PVPlacement*>(pv1_) ; 

    int nidx0_ = tree->get_nidx(pv0) ; 
    int nidx1_ = tree->get_nidx(pv1) ; 

    assert( nidx0_ == nidx0 ); 
    assert( nidx1_ == nidx1 ); 

    int cp0_ = pv0 ? pv0->GetCopyNo() : -1 ; 
    int cp1_ = pv1 ? pv1->GetCopyNo() : -1 ; 

    int cp0 = tree->get_pv_copyno(nidx0); 
    int cp1 = tree->get_pv_copyno(nidx1); 

    assert( cp0_ == cp0 ); 
    assert( cp1_ == cp1 ); 

    std::cout << " q_spec0 " << q_spec0 <<  " nidx0 " << nidx0 << " cp0 " << cp0 << std::endl ;  
    std::cout << " q_spec1 " << q_spec1 <<  " nidx1 " << nidx1 << " cp1 " << cp1 << std::endl ;  
    std::cout << "] test_get_pv_0 " << std::endl ; 
}

/**
test_get_pv_1
---------------

Note the irregularity in copyno increments from how the different types of 
PMTs are distributed::

    [ test_get_pv_1 
     q_soname HamamatsuR12860sMask_virtual nodes.size 5000
     i       0 nidx   70972 copyno      1 copyno - prev_copyno       2
     i       1 nidx   70993 copyno      4 copyno - prev_copyno       3
     i       2 nidx   71021 copyno      8 copyno - prev_copyno       4
     i       3 nidx   71042 copyno     11 copyno - prev_copyno       3
     i       4 nidx   71070 copyno     15 copyno - prev_copyno       4
     i       5 nidx   71091 copyno     18 copyno - prev_copyno       3
     i       6 nidx   71119 copyno     22 copyno - prev_copyno       4
     i       7 nidx   71140 copyno     25 copyno - prev_copyno       3
     i       8 nidx   71168 copyno     29 copyno - prev_copyno       4
     i       9 nidx   71189 copyno     32 copyno - prev_copyno       3
     ... 
     i    4991 nidx  194039 copyno  17582 copyno - prev_copyno       3
     i    4992 nidx  194067 copyno  17586 copyno - prev_copyno       4
     i    4993 nidx  194088 copyno  17589 copyno - prev_copyno       3
     i    4994 nidx  194116 copyno  17593 copyno - prev_copyno       4
     i    4995 nidx  194137 copyno  17596 copyno - prev_copyno       3
     i    4996 nidx  194165 copyno  17600 copyno - prev_copyno       4
     i    4997 nidx  194186 copyno  17603 copyno - prev_copyno       3
     i    4998 nidx  194214 copyno  17607 copyno - prev_copyno       4
     i    4999 nidx  194235 copyno  17610 copyno - prev_copyno       3
    ] test_get_pv_1 
    [ test_get_pv_1 
     q_soname NNVTMCPPMTsMask_virtual nodes.size 12612
     i       0 nidx   70965 copyno      0 copyno - prev_copyno       1
     i       1 nidx   70979 copyno      2 copyno - prev_copyno       2
     i       2 nidx   70986 copyno      3 copyno - prev_copyno       1
     i       3 nidx   71000 copyno      5 copyno - prev_copyno       2
     i       4 nidx   71007 copyno      6 copyno - prev_copyno       1
     i       5 nidx   71014 copyno      7 copyno - prev_copyno       1
     i       6 nidx   71028 copyno      9 copyno - prev_copyno       2
     i       7 nidx   71035 copyno     10 copyno - prev_copyno       1
     i       8 nidx   71049 copyno     12 copyno - prev_copyno       2
     i       9 nidx   71056 copyno     13 copyno - prev_copyno       1
     ... 
     i   12603 nidx  194158 copyno  17599 copyno - prev_copyno       1
     i   12604 nidx  194172 copyno  17601 copyno - prev_copyno       2
     i   12605 nidx  194179 copyno  17602 copyno - prev_copyno       1
     i   12606 nidx  194193 copyno  17604 copyno - prev_copyno       2
     i   12607 nidx  194200 copyno  17605 copyno - prev_copyno       1
     i   12608 nidx  194207 copyno  17606 copyno - prev_copyno       1
     i   12609 nidx  194221 copyno  17608 copyno - prev_copyno       2
     i   12610 nidx  194228 copyno  17609 copyno - prev_copyno       1
     i   12611 nidx  194242 copyno  17611 copyno - prev_copyno       2
    ] test_get_pv_1 
    [ test_get_pv_1 
     q_soname mask_PMT_20inch_vetosMask_virtual nodes.size 2400
     i       0 nidx  322253 copyno  30000 copyno - prev_copyno   30001
     i       1 nidx  322259 copyno  30001 copyno - prev_copyno       1
     i       2 nidx  322265 copyno  30002 copyno - prev_copyno       1
     i       3 nidx  322271 copyno  30003 copyno - prev_copyno       1
     i       4 nidx  322277 copyno  30004 copyno - prev_copyno       1
     i       5 nidx  322283 copyno  30005 copyno - prev_copyno       1
     i       6 nidx  322289 copyno  30006 copyno - prev_copyno       1
     i       7 nidx  322295 copyno  30007 copyno - prev_copyno       1
     i       8 nidx  322301 copyno  30008 copyno - prev_copyno       1
     i       9 nidx  322307 copyno  30009 copyno - prev_copyno       1
     ... 
     i    2391 nidx  336599 copyno  32391 copyno - prev_copyno       1
     i    2392 nidx  336605 copyno  32392 copyno - prev_copyno       1
     i    2393 nidx  336611 copyno  32393 copyno - prev_copyno       1
     i    2394 nidx  336617 copyno  32394 copyno - prev_copyno       1
     i    2395 nidx  336623 copyno  32395 copyno - prev_copyno       1
     i    2396 nidx  336629 copyno  32396 copyno - prev_copyno       1
     i    2397 nidx  336635 copyno  32397 copyno - prev_copyno       1
     i    2398 nidx  336641 copyno  32398 copyno - prev_copyno       1
     i    2399 nidx  336647 copyno  32399 copyno - prev_copyno       1
    ] test_get_pv_1 
    [ test_get_pv_1 
     q_soname PMT_3inch_pmt_solid nodes.size 25600
     i       0 nidx  194249 copyno 300000 copyno - prev_copyno  300001
     i       1 nidx  194254 copyno 300001 copyno - prev_copyno       1
     i       2 nidx  194259 copyno 300002 copyno - prev_copyno       1
     i       3 nidx  194264 copyno 300003 copyno - prev_copyno       1
     i       4 nidx  194269 copyno 300004 copyno - prev_copyno       1
     i       5 nidx  194274 copyno 300005 copyno - prev_copyno       1
     i       6 nidx  194279 copyno 300006 copyno - prev_copyno       1
     i       7 nidx  194284 copyno 300007 copyno - prev_copyno       1
     i       8 nidx  194289 copyno 300008 copyno - prev_copyno       1
     i       9 nidx  194294 copyno 300009 copyno - prev_copyno       1
     ... 
     i   25591 nidx  322204 copyno 325591 copyno - prev_copyno       1
     i   25592 nidx  322209 copyno 325592 copyno - prev_copyno       1
     i   25593 nidx  322214 copyno 325593 copyno - prev_copyno       1
     i   25594 nidx  322219 copyno 325594 copyno - prev_copyno       1
     i   25595 nidx  322224 copyno 325595 copyno - prev_copyno       1
     i   25596 nidx  322229 copyno 325596 copyno - prev_copyno       1
     i   25597 nidx  322234 copyno 325597 copyno - prev_copyno       1
     i   25598 nidx  322239 copyno 325598 copyno - prev_copyno       1
     i   25599 nidx  322244 copyno 325599 copyno - prev_copyno       1
    ] test_get_pv_1 

**/

void test_get_pv_1(const U4Tree* tree, const char* q_soname)
{
    std::cout << "[ test_get_pv_1 " << std::endl ; 

    unsigned edge = SSys::getenvunsigned("EDGE", 10); 

    std::vector<int> nodes ; 
    tree->st->find_lvid_nodes(nodes, q_soname );  
    std::cout << " q_soname " << q_soname << " nodes.size " << nodes.size() << std::endl ; 

    int copyno = -1 ; 
    int prev_copyno = -1 ; 
    for(unsigned i=0 ; i < nodes.size() ; i++)
    {
        int nidx = nodes[i] ; 
        prev_copyno = copyno ;  
        copyno = tree->get_pv_copyno(nidx);  

        int copyno_0 = tree->st->get_copyno(nidx); 
        assert( copyno == copyno_0 ); 

        if( i < edge || (i > (nodes.size() - edge))) 
            std::cout 
                << " i " << std::setw(7) << i
                << " nidx " << std::setw(7) << nidx 
                << " copyno " << std::setw(6) << copyno 
                << " copyno - prev_copyno  " << std::setw(6) << (copyno - prev_copyno)  
                << std::endl 
                ; 
        else if( i == edge ) 
            std::cout 
                << " ... " << std::endl ; 

    }
    std::cout << "] test_get_pv_1 " << std::endl ; 
}

void test_get_pv_1(const U4Tree* tree)
{
    std::vector<std::string> sub_sonames ; 
    tree->st->get_sub_sonames(sub_sonames); 
  
    for(unsigned i=0 ; i < sub_sonames.size() ; i++)
    {
        const char* soname = sub_sonames[i].c_str();  
        test_get_pv_1(tree, soname ); 
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = SPath::Resolve("$GDMLPath", NOOP ) ; 
    LOG(info) << " path [" << path << "]" ; 
    LOG_IF(fatal, path == nullptr) << " $GDMLPath null, see SOpticksResource  " ; 
    if( path == nullptr ) return 0 ; 


    const G4VPhysicalVolume* world = U4GDML::Read(path) ;  

    stree* st = new stree ; 
    U4Tree* tree = U4Tree::Create(st, world) ; 
    st->save(FOLD); 

    //test_get_pv_0(tree); 
    test_get_pv_1(tree); 

    return 0 ;  
}
