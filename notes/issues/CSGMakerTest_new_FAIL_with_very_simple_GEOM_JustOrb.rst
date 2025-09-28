CSGMakerTest_new_FAIL_with_very_simple_GEOM_JustOrb
=====================================================

Reason for the FAIL now is recent addition of stree::save_desc to SSim::save

FIXED by adding proptection against invalid nidx in stree.h



::

    Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2025-09-28 19:16:19.386 INFO  [3178725] [GetNames@45]  names.size 34
    2025-09-28 19:16:19.386 INFO  [3178725] [main@66] JustOrb
    [Detaching after vfork from child process 3178728]
    2025-09-28 19:16:19.398 INFO  [3178725] [CSGMaker::MakeGeom@1457]  so 0x7ffff61b6010
    2025-09-28 19:16:19.399 INFO  [3178725] [CSGMaker::MakeGeom@1458]  so.desc CSGSolid          JustOrb primNum/Offset     1    0 ce ( 0.000, 0.000, 0.000,100.000) 
    2025-09-28 19:16:19.399 INFO  [3178725] [CSGMaker::MakeGeom@1459]  fd.desc CSGFoundry  num_total 1 num_solid 1 num_prim 1 num_node 1 num_plan 0 num_tran 1 num_itra 1 num_inst 1 gas 0 meshname 1 mmlabel 1 mtime 1759058179 mtimestamp 20250928_191619 sim Y
    2025-09-28 19:16:19.399 INFO  [3178725] [main@69] CSGFoundry  num_total 1 num_solid 1 num_prim 1 num_node 1 num_plan 0 num_tran 1 num_itra 1 num_inst 1 gas 0 meshname 1 mmlabel 1 mtime 1759058179 mtimestamp 20250928_191619 sim Y
    [stree::save_desc_ fold [/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry/SSim/stree/desc]
    -stree::save_desc_ name [NRT.txt]
    -stree::save_desc_ name [_csg.txt]
    -stree::save_desc_ name [bd.txt]
    -stree::save_desc_ name [factor.txt]
    -stree::save_desc_ name [inst.txt]
    -stree::save_desc_ name [inst_info.txt]
    -stree::save_desc_ name [inst_info_check.txt]
    -stree::save_desc_ name [lvid.txt]
    -stree::save_desc_ name [material.txt]
    -stree::save_desc_ name [mesh.txt]
    -stree::save_desc_ name [meta.txt]
    -stree::save_desc_ name [mt.txt]
    -stree::save_desc_ name [nds.txt]
    -stree::save_desc_ name [node_EBOUNDARY.txt]
    -stree::save_desc_ name [node_ECOPYNO.txt]
    -stree::save_desc_ name [node_ELVID.txt]
    -stree::save_desc_ name [node_solids.txt]
    -stree::save_desc_ name [nodes.txt]

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff73a13a2 in stree::get_children (this=0x4616f0, children=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1255
    1255	    assert( nd.index == nidx );
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64
    (gdb) bt
    #0  0x00007ffff73a13a2 in stree::get_children (this=0x4616f0, children=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1255
    #1  0x00007ffff73a14d4 in stree::get_progeny (this=0x4616f0, progeny=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1271
    #2  0x00007ffff73a15de in stree::desc_progeny[abi:cxx11](int, int) const (this=0x4616f0, nidx=0, edge=1000) at /home/blyth/opticks/sysrap/stree.h:1280
    #3  0x00007ffff739e24d in stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}::operator()[abi:cxx11]() const (__closure=0x4727f0)
        at /home/blyth/opticks/sysrap/stree.h:923
    #4  0x00007ffff73ddbd4 in std::__invoke_impl<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&>(std::__invoke_other, stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&) (__f=...)
        at /usr/include/c++/11/bits/invoke.h:61
    #5  0x00007ffff73d8605 in std::__invoke_r<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&>(stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&) (__fn=...)
        at /usr/include/c++/11/bits/invoke.h:116
    #6  0x00007ffff73d0e99 in std::_Function_handler<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (), stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}>::_M_invoke(std::_Any_data const&) (__functor=...) at /usr/include/c++/11/bits/std_function.h:291
    #7  0x00007ffff73bcec3 in std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>::operator()() const (this=0x4727f0) at /usr/include/c++/11/bits/std_function.h:590
    #8  0x00007ffff739df92 in stree::save_desc_ (this=0x4616f0, fold=0x4725c0 "/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry/SSim/stree/desc") at /home/blyth/opticks/sysrap/stree.h:902
    #9  0x00007ffff739ddc9 in stree::save_desc (this=0x4616f0, base=0x4702c0 "/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry/SSim", midl=0x7ffff74c264b "stree") at /home/blyth/opticks/sysrap/stree.h:884
    #10 0x00007ffff738f6a6 in SSim::save (this=0x4611b0, base=0x462fc0 "/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry", reldir=0x7ffff7da86a8 "SSim") at /home/blyth/opticks/sysrap/SSim.cc:438
    #11 0x00007ffff7c4e341 in CSGFoundry::save_ (this=0x462cd0, dir_=0x465be0 "$TMP/CSGMakerTest/JustOrb/CSGFoundry") at /home/blyth/opticks/CSG/CSGFoundry.cc:2699
    #12 0x00007ffff7c4d755 in CSGFoundry::save (this=0x462cd0, base=0x462f70 "$TMP/CSGMakerTest/JustOrb", rel=0x7ffff7da84e4 "CSGFoundry") at /home/blyth/opticks/CSG/CSGFoundry.cc:2626
    #13 0x00000000004061b5 in main (argc=1, argv=0x7fffffffb688) at /home/blyth/opticks/CSG/tests/CSGMakerTest.cc:73
    (gdb) 


    (gdb) f 0
    #0  0x00007ffff73a13a2 in stree::get_children (this=0x4616f0, children=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1255
    1255	    assert( nd.index == nidx );
    (gdb) p nd
    $1 = (const snode &) <error reading variable: Cannot access memory at address 0x0>
    (gdb) p nidx
    $2 = 0
    (gdb) 



HMM must be missing some protection::

     49 int main(int argc, char** argv)
     50 {
     51      const char* arg =  argc > 1 ? argv[1] : nullptr ;
     52      bool listnames = arg && ( strcmp(arg,"N") == 0 || strcmp(arg,"n") == 0 ) ;
     53      OPTICKS_LOG(argc, argv);
     54 
     55      SSim* sim = SSim::Create();
     56      assert(sim);
     57      if(!sim) std::raise(SIGINT);
     58 
     59      std::vector<std::string> names ;
     60      GetNames(names, listnames);
     61      if(listnames) return 0 ;
     62 
     63      for(unsigned i=0 ; i < names.size() ; i++)
     64      {
     65          const char* name = names[i].c_str() ;
     66          LOG(info) << name ;
     67 
     68          CSGFoundry* fd = CSGMaker::MakeGeom( name );
     69          LOG(info) << fd->desc();
     70 
     71          const char* base = spath::Join("$TMP/CSGMakerTest",name) ;
     72 
     73          fd->save(base);
     74 
     75          CSGFoundry* lfd = CSGFoundry::Load(base);
     76 
     77 
     78          LOG(info) << " lfd.loaddir " << lfd->loaddir ;
     79 
     80          int rc = CSGFoundry::Compare(fd, lfd );
     81          assert( 0 == rc );
     82          if(0!=rc) std::raise(SIGINT);
     83      }
     84 
     85      return 0 ;
     86 }




    -stree::save_desc_ name [progeny.txt]

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff73a13a2 in stree::get_children (this=0x4616f0, children=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1256
    1256	    bool nidx_expect = nd.index == nidx ;
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64
    (gdb) bt
    #0  0x00007ffff73a13a2 in stree::get_children (this=0x4616f0, children=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1256
    #1  0x00007ffff73a1574 in stree::get_progeny (this=0x4616f0, progeny=..., nidx=0) at /home/blyth/opticks/sysrap/stree.h:1280
    #2  0x00007ffff73a167e in stree::desc_progeny[abi:cxx11](int, int) const (this=0x4616f0, nidx=0, edge=1000) at /home/blyth/opticks/sysrap/stree.h:1289
    #3  0x00007ffff739e24d in stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}::operator()[abi:cxx11]() const (__closure=0x4727f0)
        at /home/blyth/opticks/sysrap/stree.h:924
    #4  0x00007ffff73ddc74 in std::__invoke_impl<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&>(std::__invoke_other, stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&) (__f=...)
        at /usr/include/c++/11/bits/invoke.h:61
    #5  0x00007ffff73d86a5 in std::__invoke_r<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&>(stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}&) (__fn=...)
        at /usr/include/c++/11/bits/invoke.h:116
    #6  0x00007ffff73d0f39 in std::_Function_handler<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (), stree::populate_descMap(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()> > > >&) const::{lambda()#7}>::_M_invoke(std::_Any_data const&) (__functor=...) at /usr/include/c++/11/bits/std_function.h:291
    #7  0x00007ffff73bcf63 in std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > ()>::operator()() const (this=0x4727f0) at /usr/include/c++/11/bits/std_function.h:590
    #8  0x00007ffff739dfdd in stree::save_desc_ (this=0x4616f0, fold=0x4725c0 "/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry/SSim/stree/desc") at /home/blyth/opticks/sysrap/stree.h:904
    #9  0x00007ffff739ddc9 in stree::save_desc (this=0x4616f0, base=0x4702c0 "/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry/SSim", midl=0x7ffff74c2673 "stree") at /home/blyth/opticks/sysrap/stree.h:884
    #10 0x00007ffff738f6a6 in SSim::save (this=0x4611b0, base=0x462fc0 "/data1/blyth/tmp/CSGMakerTest/JustOrb/CSGFoundry", reldir=0x7ffff7da86a8 "SSim") at /home/blyth/opticks/sysrap/SSim.cc:438
    #11 0x00007ffff7c4e341 in CSGFoundry::save_ (this=0x462cd0, dir_=0x465be0 "$TMP/CSGMakerTest/JustOrb/CSGFoundry") at /home/blyth/opticks/CSG/CSGFoundry.cc:2699
    #12 0x00007ffff7c4d755 in CSGFoundry::save (this=0x462cd0, base=0x462f70 "$TMP/CSGMakerTest/JustOrb", rel=0x7ffff7da84e4 "CSGFoundry") at /home/blyth/opticks/CSG/CSGFoundry.cc:2626
    #13 0x00000000004061b5 in main (argc=1, argv=0x7fffffffb688) at /home/blyth/opticks/CSG/tests/CSGMakerTest.cc:73
    (gdb) 



