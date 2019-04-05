Ubuntu16.04.6-gcc5.4.0-npy-NNodeDumpTest-SEGV
=================================================



Down to 1 expected fail with the workaround
-----------------------------------------------

::

    99% tests passed, 1 tests failed out of 118

    Total Test time (real) =   1.66 sec

    The following tests FAILED:
         83 - NPYTest.NLoadTest (Child aborted)
    Errors while running CTest
    Fri Apr  5 16:37:09 CST 2019
    blyth@blyth-VirtualBox:~/opticks/npy$ NLoadTest 
    2019-04-05 16:37:23.788 INFO  [2290] [NPY<T>::load@633] NPY<T>::load /usr/local/opticks/opticksdata/gensteps/dayabay/cerenkov/./1.npy
    2019-04-05 16:37:23.797 WARN  [2290] [NPY<T>::load@658] NPY<T>::load failed for path [/usr/local/opticks/opticksdata/gensteps/dayabay/cerenkov/./1.npy] use debugload to see why
    2019-04-05 16:37:23.798 INFO  [2290] [NPY<T>::load@633] NPY<T>::load /usr/local/opticks/opticksdata/gensteps/juno/cerenkov/./1.npy
    2019-04-05 16:37:23.798 WARN  [2290] [NPY<T>::load@658] NPY<T>::load failed for path [/usr/local/opticks/opticksdata/gensteps/juno/cerenkov/./1.npy] use debugload to see why
    2019-04-05 16:37:23.798 INFO  [2290] [NPY<T>::load@633] NPY<T>::load /usr/local/opticks/opticksdata/gensteps/dayabay/scintillation/./1.npy
    2019-04-05 16:37:23.799 WARN  [2290] [NPY<T>::load@658] NPY<T>::load failed for path [/usr/local/opticks/opticksdata/gensteps/dayabay/scintillation/./1.npy] use debugload to see why
    2019-04-05 16:37:23.799 INFO  [2290] [NPY<T>::load@633] NPY<T>::load /usr/local/opticks/opticksdata/gensteps/juno/scintillation/./1.npy
    2019-04-05 16:37:23.799 WARN  [2290] [NPY<T>::load@658] NPY<T>::load failed for path [/usr/local/opticks/opticksdata/gensteps/juno/scintillation/./1.npy] use debugload to see why
    NLoadTest: /home/blyth/opticks/npy/tests/NLoadTest.cc:21: int main(int, char**): Assertion `gs_0' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/npy$ 


    blyth@blyth-VirtualBox:~/opticks/npy$ uname -a
    Linux blyth-VirtualBox 4.15.0-45-generic #48~16.04.1-Ubuntu SMP Tue Jan 29 18:03:48 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
    blyth@blyth-VirtualBox:~/opticks/npy$ gcc -v
    Using built-in specs.
    COLLECT_GCC=gcc
    COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-linux-gnu/5/lto-wrapper
    Target: x86_64-linux-gnu
    Configured with: ../src/configure -v --with-pkgversion='Ubuntu 5.4.0-6ubuntu1~16.04.11' --with-bugurl=file:///usr/share/doc/gcc-5/README.Bugs --enable-languages=c,ada,c++,java,go,d,fortran,objc,obj-c++ --prefix=/usr --program-suffix=-5 --enable-shared --enable-linker-build-id --libexecdir=/usr/lib --without-included-gettext --enable-threads=posix --libdir=/usr/lib --enable-nls --with-sysroot=/ --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --with-default-libstdcxx-abi=new --enable-gnu-unique-object --disable-vtable-verify --enable-libmpx --enable-plugin --with-system-zlib --disable-browser-plugin --enable-java-awt=gtk --enable-gtk-cairo --with-java-home=/usr/lib/jvm/java-1.5.0-gcj-5-amd64/jre --enable-java-home --with-jvm-root-dir=/usr/lib/jvm/java-1.5.0-gcj-5-amd64 --with-jvm-jar-dir=/usr/lib/jvm-exports/java-1.5.0-gcj-5-amd64 --with-arch-directory=amd64 --with-ecj-jar=/usr/share/java/eclipse-ecj.jar --enable-objc-gc --enable-multiarch --disable-werror --with-arch-32=i686 --with-abi=m64 --with-multilib-list=m32,m64,mx32 --enable-multilib --with-tune=generic --enable-checking=release --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu
    Thread model: posix
    gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.11) 
    blyth@blyth-VirtualBox:~/opticks/npy$ 




Possible Workaround for the issue 
-----------------------------------

Workaround (it seems so far) is to rework the nnode primitives to avoid 
relying on the implicit copy ctor for their creation.

That means no more::

    nsphere* sph = new nsphere(make_sphere(0,0,0, 10)) ;

Now directly onto heap::

    nsphere* sph = make_sphere(0,0,0, 10) ;


::


    -void nnode::Init( nnode& n , OpticksCSG_t type, nnode* left, nnode* right )
    +void nnode::Init( nnode* n , OpticksCSG_t type, nnode* left, nnode* right )
     {
    -    n.idx = 0 ; 
    -    n.type = type ; 
    +    n->idx = 0 ; 
    +    n->type = type ; 
     

::

    -inline nnode nnode::make_node(OpticksCSG_t operator_, nnode* left, nnode* right )
    +inline nnode* nnode::make_node(OpticksCSG_t operator_, nnode* left, nnode* right )
     {
    -    nnode n ;    nnode::Init(n, operator_ , left, right ); return n ;
    +    nnode* n = new nnode ;    nnode::Init(n, operator_ , left, right ); return n ;
     }
     
     struct NPY_API nunion : nnode {
         float operator()(float x, float y, float z) const ;
    -    static nunion make_union(nnode* left=NULL, nnode* right=NULL);
    +    static nunion* make_union(nnode* left=NULL, nnode* right=NULL);
     };
     struct NPY_API nintersection : nnode {
         float operator()(float x, float y, float z) const ;
    -    static nintersection make_intersection(nnode* left=NULL, nnode* right=NULL);
    +    static nintersection* make_intersection(nnode* left=NULL, nnode* right=NULL);
     };
     struct NPY_API ndifference : nnode {
         float operator()(float x, float y, float z) const ;
    -    static ndifference make_difference(nnode* left=NULL, nnode* right=NULL);
    +    static ndifference* make_difference(nnode* left=NULL, nnode* right=NULL);
     };
     
    -inline nunion nunion::make_union(nnode* left, nnode* right)
    +inline nunion* nunion::make_union(nnode* left, nnode* right)
     {
    -    nunion n ;         nnode::Init(n, CSG_UNION , left, right ); return n ; 
    +    nunion* n = new nunion ;         nnode::Init(n, CSG_UNION , left, right ); return n ; 
     }
    -inline nintersection nintersection::make_intersection(nnode* left, nnode* right)
    +inline nintersection* nintersection::make_intersection(nnode* left, nnode* right)
     {
    -    nintersection n ;  nnode::Init(n, CSG_INTERSECTION , left, right ); return n ;
    +    nintersection* n = new nintersection ;  nnode::Init(n, CSG_INTERSECTION , left, right ); return n ;
     }
    -inline ndifference ndifference::make_difference(nnode* left, nnode* right)
    +inline ndifference* ndifference::make_difference(nnode* left, nnode* right)
     {
    -    ndifference n ;    nnode::Init(n, CSG_DIFFERENCE , left, right ); return n ;
    +    ndifference* n = new ndifference ;    nnode::Init(n, CSG_DIFFERENCE , left, right ); return n ;
     }





Investigate on virtualbox+Ubuntu16.04.6 (gcc 5.4.0) 
------------------------------------------------------

See npy/tests/NNodeDumpMinimalTest.cc especially::


::

    void t1c()  // works 
    {
        LOG(info); 
        nsphere* o = make_sphere(0.f,0.f,-50.f,100.f);
        nnode* n = o ; 
        n->dump();
    }


    void t1d()  // fails : so the problem is related to the original object going out of scope : somehow handled different in gcc 5.4.0
    {
        LOG(info); 

        nsphere* a = NULL ; 
        {
            nsphere o = make_sphere(0.f,0.f,-50.f,100.f);

            // why should o going out of scope matter ? 
            // perhaps implicit copy ctor is being overly lazy : overly agressive optimization  ??
            // :google:`gcc 5.4 optimization bug` 

            a = new nsphere(o) ;  // implicit copy ctor  
        }
        nnode* n = a ; 
        n->dump();
    }



Isolating the issue
-----------------------

Mail of Thu April 4, 2019::


    Hi Elias,

    I succeeded to reproduce what looks like the issue you are seeing by installing
    virtualbox+Ubuntu16.04.6 (gcc 5.4.0) and doing a partial Opticks install.
    Has to be partial as I think CUDA doesnt work from virtualbox.

    Interestingly trying with virtualbox+Ubuntu18.04.2 (gcc 7.3.0) does not have the
    issue.    Also no such problem on macOS High Sierra (llvm 9.0.0) or Centos 7 (gcc 4.8.5)

    Do you see the same three failing as below ?

    blyth@blyth-VirtualBox:~/opticks/npy$ om-test
    === om-test-one : npy             /home/blyth/opticks/npy                                      /usr/local/opticks/build/npy                                
    Thu Apr  4 22:14:19 CST 2019
    ...
    97% tests passed, 3 tests failed out of 117

    Total Test time (real) =   1.92 sec

    The following tests FAILED:
         67 - NPYTest.NNodeDumpTest (SEGFAULT)
         82 - NPYTest.NLoadTest (Child aborted)
         92 - NPYTest.NCSGRoundTripTest (SEGFAULT)
    Errors while running CTest


    Still no clue as to the cause, but at least the scope of the issue is narrowed :
    and I can dissect it directly.
    See my last few commits up to the below for the details.
         https://bitbucket.org/simoncblyth/opticks/commits/3cde87d6f4ebb95754d2407ec655380c7e400fe9

    Especially
        bin/vbx.bash
       notes/issues/Ubuntu16.04.6-gcc5.4.0-npy-NNodeDumpTest-SEGV.rst

    I tried switching from reference to pointer in NNodeDump2 but it makes no difference.

    Simon





Table of Ubuntu release dates: https://wiki.ubuntu.com/Releases


::

    97% tests passed, 3 tests failed out of 117

    Total Test time (real) =   1.93 sec

    The following tests FAILED:
         67 - NPYTest.NNodeDumpTest (SEGFAULT)
         82 - NPYTest.NLoadTest (Child aborted)
         92 - NPYTest.NCSGRoundTripTest (SEGFAULT)
    Errors while running CTest
    Thu Apr  4 21:19:59 CST 2019


::

    (gdb) r
    Starting program: /usr/local/opticks/lib/NNodeDumpTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
    2019-04-04 21:20:41.886 INFO  [3247] [test_dump@15] 
     sample idx : 0

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7972274 in NNodeDump2::dump_label (this=0x621710, pfx=0x7ffff7ac01c2 "du") at /home/blyth/opticks/npy/NNodeDump2.cpp:38
    38           << std::setw(nnode::desc_indent) << m_node->desc() 
    (gdb) list
    33  
    34  void NNodeDump2::dump_label(const char* pfx) const 
    35  {
    36      std::cout 
    37           << std::setw(3) << (  pfx ? pfx : "-" ) << " " 
    38           << std::setw(nnode::desc_indent) << m_node->desc() 
    39           ; 
    40  }
    41  
    42  void NNodeDump2::dump_base() const 
    (gdb) p m_node
    $1 = (const nnode *) 0x7fffffffe0c0
    (gdb) p m_node->desc()
    Cannot access memory at address 0x40
    (gdb) p m_node
    $2 = (const nnode *) 0x7fffffffe0c0
    (gdb) bt
    #0  0x00007ffff7972274 in NNodeDump2::dump_label (this=0x621710, pfx=0x7ffff7ac01c2 "du") at /home/blyth/opticks/npy/NNodeDump2.cpp:38
    #1  0x00007ffff797237b in NNodeDump2::dump_base (this=0x621710) at /home/blyth/opticks/npy/NNodeDump2.cpp:44
    #2  0x00007ffff7972105 in NNodeDump2::dump (this=0x621710) at /home/blyth/opticks/npy/NNodeDump2.cpp:22
    #3  0x00007ffff795de5d in nnode::dump (this=0x621730, msg=0x0) at /home/blyth/opticks/npy/NNode.cpp:1360
    #4  0x000000000040492c in test_dump (nodes=std::vector of length 12, capacity 16 = {...}, idx=0) at /home/blyth/opticks/npy/tests/NNodeDumpTest.cc:16
    #5  0x00000000004049a8 in test_dump (nodes=std::vector of length 12, capacity 16 = {...}) at /home/blyth/opticks/npy/tests/NNodeDumpTest.cc:21
    #6  0x0000000000404b6d in main (argc=1, argv=0x7fffffffe388) at /home/blyth/opticks/npy/tests/NNodeDumpTest.cc:38
    (gdb) f 3
    #3  0x00007ffff795de5d in nnode::dump (this=0x621730, msg=0x0) at /home/blyth/opticks/npy/NNode.cpp:1360
    1360        _dump->dump();
    (gdb) 


