sn_ctor_dtor_pool
====================

sn_test.cc::

    189 int main(int argc, char** argv)
    190 {       
    200     test_CommonTree_1(3);    // segv that doesnt always happen under debugger
    201     //test_CommonTree_1(4); 
    202     //test_CommonTree_1(8); 
    203     
    204     return 0 ;
    205 }


::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=EXC_I386_GPFLT)
      * frame #0: 0x0000000100014aba sn_test`sn::num_node_r(this=0x6478616d20322020, d=2) const at sn.h:297
        frame #1: 0x0000000100014ade sn_test`sn::num_node_r(this=0x00000001004000e0, d=1) const at sn.h:297
        frame #2: 0x0000000100014b1d sn_test`sn::num_node_r(this=0x0000000100400350, d=0) const at sn.h:298
        frame #3: 0x0000000100014a07 sn_test`sn::num_node(this=0x0000000100400350) const at sn.h:292
        frame #4: 0x000000010000df4c sn_test`sn::desc(this=0x0000000100400350) const at sn.h:486
        frame #5: 0x000000010000d35d sn_test`test_CommonTree_1(num_leaves=3) at sn_test.cc:61
        frame #6: 0x000000010000fca2 sn_test`main(argc=1, argv=0x00007ffeefbfe9d0) at sn_test.cc:200
        frame #7: 0x00007fff55514015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x000000010000df4c sn_test`sn::desc(this=0x0000000100400350) const at sn.h:486
       483 	    ss << "sn::desc"
       484 	       << " pid " << std::setw(3) << pid
       485 	       << " t " << std::setw(3) << t 
    -> 486 	       << " num_node " << std::setw(3) << num_node() 
       487 	       << " num_leaf " << std::setw(3) << num_leaf() 
       488 	       << " maxdepth " << std::setw(2) << maxdepth() 
       489 	       << " is_positive_form " << ( is_positive_form() ? "Y" : "N" ) 
    (lldb) f 5
    frame #5: 0x000000010000d35d sn_test`test_CommonTree_1(num_leaves=3) at sn_test.cc:61
       58  	    std::cout << sn::Desc(); 
       59  	
       60  	    std::cout << "[ delete root  root->pid " << root->pid << std::endl ; 
    -> 61  	    std::cout << " root->desc " << root->desc() << std::endl ; 
       62  	
       63  	    delete root ; 
       64  	    std::cout << "]" << std::endl ; 
    (lldb) f 3
    frame #3: 0x0000000100014a07 sn_test`sn::num_node(this=0x0000000100400350) const at sn.h:292
       289 	
       290 	inline int sn::num_node() const
       291 	{
    -> 292 	    return num_node_r(0);
       293 	}
       294 	inline int sn::num_node_r(int d) const
       295 	{
    (lldb) f 2
    frame #2: 0x0000000100014b1d sn_test`sn::num_node_r(this=0x0000000100400350, d=0) const at sn.h:298
       295 	{
       296 	    int nn = 1 ;   // always at least 1 node,  no exclusion of CSG_ZERO
       297 	    nn += l ? l->num_node_r(d+1) : 0 ; 
    -> 298 	    nn += r ? r->num_node_r(d+1) : 0 ; 
       299 	    return nn ;
       300 	}
       301 	
    (lldb) p l
    (sn *) $0 = 0x0000000100400140
    (lldb) p r
    (sn *) $1 = 0x00000001004000e0
    (lldb) p *l
    (sn) $2 = {
      pid = 2
      t = 1
      depth = 0
      subdepth = 0
      complement = false
      l = 0x0000000100400410
      r = 0x0000000100400080
    }
    (lldb) p *r
    (sn) $3 = {
      pid = 1970151473
      t = 1869504365
      depth = 538994020
      subdepth = 1847604000
      complement = true
      l = 0x6478616d20322020
      r = 0x2031202068747065
    }
    (lldb) 


