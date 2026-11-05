review/csg_nd_ultra_compact
============================

What is needed within an ultra-compact csg node used with an RPN stream ? Aim for 32 bits


cn.h
-----

::

    OPER typecode[6] + leftspan[?] + rightspan[?]                          (32-6)/2 = 13 if nothing else needed?
    LIST typecode[6] + childCount 
    PRIM typecode[6] + paramIdx[13] + gtranIdx[14] + complement[1]         6+13+14+1=34 
          [HMM put gtranIdx in param - float param needed when also need gtranIdx]
                                             
* no problem for OPER and LIST nodes, but PRIM is a squeeze when include gTran
* total node counts are low ~1729 - could handle gtran+bbox with mirror buffers having same item counts
  as the nodes


::

     In [22]: 5+13+13+1
     Out[22]: 32

::

    In [23]: 0x1fff   ## uniqing transforms would allow to squeeze gtranIdx into 13 bits
    Out[23]: 8191


* HMM: how much will putting gtranIdx into param will increase the param size ?


* HMM : I CHOOSE BIT LAYOUT SUITABLE FOR JUNO GEOMETRY
* THEN I CAN MAKE RPN MODE OPTIONAL - WITH ASSERTS THAT GEOMETRY COUNTS FIT IN THE BIT WIDTH CHOICES 


* all nodes need typecode [6 bits] (sysrap/OpticksCSG.h)

  * With some shuffling in OpticksCSG.h could get down to 6 bits easily::

        In [4]: 0x3f
        Out[4]: 63

  * 5 bits also possible, with more pain - and no contingency::

        In [5]: 0x1f
        Out[5]: 31


* operator nodes need : left span + right span  

* list node need : childCount

* primitive nodes need : paramIdx [13bits]

  * avoid float params in the node - by refering to those in rpn_param buffer

::

    In [13]: 0x3fff
    Out[13]: 16383

    In [14]: 0x1fff
    Out[14]: 8191


::


    CMDLINE:/tmp/blyth/local/opticks_Release_g411/bin/f.py
    f.base:.

      : f.sn                                               :           (1729, 19) : 59 days, 0:41:02.212898 
      : f.s_bb                                             :            (1167, 6) : 59 days, 0:41:02.212898 
      : f.s_pa                                             :            (1155, 6) : 59 days, 0:41:02.212898 
      : f.s_tv                                             :      (1729, 2, 4, 4) : 59 days, 0:41:02.212898 
      : f.NPFold_index                                     :                 (4,) : 59 days, 0:41:02.212898 
      : f.NPFold_names                                     :                 (0,) : 59 days, 0:41:02.212898 

     min_stamp : 2026-09-07 14:59:55.764472 
     max_stamp : 2026-09-07 14:59:55.764472 
     dif_stamp : 0:00:00 
     age_stamp : 59 days, 0:41:02.212898 

    In [1]: pwd
    Out[1]: '/home/blyth/junosw/InstallArea/Debug_g411_main/.opticks/GEOM/J26_4_1_opticks_Debug_g411/CSGFoundry/SSim/stree/_csg'



gtranIdx : only used for leaf ?
--------------------------------------

::

    174 LEAF_FUNC
    175 void intersect_leaf(bool& valid_isect, float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction, bool dumpxyz )
    176 {
    177     valid_isect = false ;
    178     isect.x = 0.f ;
    179     isect.y = 0.f ;
    180     isect.z = 0.f ;
    181     isect.w = 0.f ;
    182 
    183     const unsigned typecode = node->typecode() ;
    184     const unsigned gtransformIdx = node->gtransformIdx() ;
    185     const bool complement = node->is_complement();
    186 
    187     const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    188 
    189     float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin ;
    190     float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction ;




::

    [lo] A[blyth@localhost CSG]$ grep gtransformIdx *.*
    CSGCopy.cc:    unsigned sTranIdx = snd->gtransformIdx(); 
    CSGFoundry.cc:            unsigned tranIdx = nd->gtransformIdx();
    CSGFoundry.cc:            unsigned o_tranIdx = ond->gtransformIdx();
    CSGFoundry.cc:            unsigned c_tranIdx2 = cnd.gtransformIdx() ;
    csg_intersect_leaf.h:    const unsigned gtransformIdx = node->gtransformIdx() ; 
    csg_intersect_leaf.h:    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    csg_intersect_leaf.h:    const unsigned gtransformIdx = node->gtransformIdx() ; 
    csg_intersect_leaf.h:    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    csg_intersect_leaf.h:    printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    csg_intersect_leaf.h:    //printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    CSGNode.cc:    unsigned trIdx = gtransformIdx();
    CSGNode.h:  * simliarly index, gtransformIdx, complement could be combined
    CSGNode.h:    |    |                |                |  typecode      | gtransformIdx  |  a.node[:,3,3].view(np.int32) & 0x7fffffff      |
    CSGNode.h:typecode gtransformIdx complement
    CSGNode.h:    gtransformIdx = cf.node.view(np.uint32)[:,3,3] & 0x7fffffff
    CSGNode.h:* TODO: complement is naturally related with typecode (not with gtransformIdx)
    CSGNode.h:gtransformIdx
    CSGNode.h:    NODE_METHOD void setComplement( bool complement ){  setTransformComplement( gtransformIdx(), complement) ; }
    CSGNode.h:    NODE_METHOD unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None
    [lo] A[blyth@localhost CSG]$ 



::

    MDLINE:/tmp/blyth/local/opticks_Release_g411/bin/f.py
    f.base:.

      : f.framespec                                        :                (30,) : 59 days, 0:32:25.692440 
      : f.meshname                                         :               (472,) : 59 days, 0:29:55.679371 
      : f.tessname                                         :               (472,) : 59 days, 0:30:38.036390 
      : f.primname                                         :              (3650,) : 59 days, 0:29:55.677371 
      : f.mmlabel                                          :                (13,) : 59 days, 0:29:55.677371 
      : f.meta                                             :                (37,) : 59 days, 0:29:55.677371 
      : f.solid                                            :           (13, 3, 4) : 59 days, 0:29:55.677371 
      : f.prim                                             :         (3650, 4, 4) : 59 days, 0:29:55.673371 
      : f.node                                             :        (29173, 4, 4) : 59 days, 0:29:55.639371 
      : f.tran                                             :         (9587, 4, 4) : 59 days, 0:29:55.626371 
      : f.itra                                             :         (9587, 4, 4) : 59 days, 0:29:55.615371 
      : f.inst                                             :        (49186, 4, 4) : 59 days, 0:29:55.562371 
      : f.SSim                                             : SUBFOLD 

     min_stamp : 2026-09-07 14:57:25.238402 
     max_stamp : 2026-09-07 14:59:55.368471 
     dif_stamp : 0:02:30.130069 
     age_stamp : 59 days, 0:29:55.562371 

    In [1]: 0xfff
    Out[1]: 4095

    In [2]: 0x1fff
    Out[2]: 8191

    In [3]: 0x3fff  # 14
    Out[3]: 16383

    In [4]: 0xffff
    Out[4]: 65535

    In [5]: 0x7fff
    Out[5]: 32767

    In [6]: 0x3fff     ## 14 bits
    Out[6]: 16383

    In [7]: 0x1fff     ## 13 bits
    Out[7]: 8191



::


    In [10]: import numpy as np

    In [11]: unique_transforms = np.unique(f.tran.reshape(f.tran.shape[0], -1), axis=0)

    In [12]: len(unique_transforms)
    Out[12]: 8084

    In [13]: rounded_tran = np.round(f.tran.reshape(f.tran.shape[0], -1), decimals=5)

    In [14]: len(rounded_tran)
    Out[14]: 9587

    In [15]: len(np.unique(rounded_tran, axis=0))
    Out[15]: 8080












