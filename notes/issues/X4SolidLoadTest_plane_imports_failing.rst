FIXED :  X4SolidLoadTest_plane_imports_failing
================================================

* FIXED : an off-by-one bug 



bug post-mortem 
------------------

Suspicion : dormant off-by-one bug surfaced in direct workflow, 
because that brought into use some ancient code ?


Checking old version of NCSG.cpp it uses the dirty in memory overwrite 
of m_nodes with the m_planes being left as they were loaded from the
python serialization.  

* https://bitbucket.org/simoncblyth/opticks/src/3aa969a393617f34374fe3654126363aa0995c60/npy/NCSG.cpp?at=default


python serialization very clearly uses 1-based for planes and transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The analytic gdml2gltf workflow lays down the 
planes and transforms with 1-based indexing in analytic/csg.py:serialize::

     669     def serialize(self, suppress_identity=False):
     670         """
     671         Array is sized for a complete tree, empty slots stay all zero
     672         """
     673         if not self.is_root: self.analyse()
     674         buf = np.zeros((self.totnodes,self.NJ,self.NK), dtype=np.float32 )
     675 
     676         transforms = []
     677         planes = []
     678 
     679         def serialize_r(node, idx):
     680             """
     681             :param node:
     682             :param idx: 0-based complete binary tree index, left:2*idx+1, right:2*idx+2 
     683             """
     684             trs = node.transform
     685             if trs is None and suppress_identity == False:
     686                 trs = np.eye(4, dtype=np.float32)
     687                 # make sure root node always has a transform, incase of global placement 
     688                 # hmm root node is just an op-node it doesnt matter, need transform slots for all primitives 
     689             pass
     690 
     691             if trs is None:
     692                 itransform = 0
     693             else:
     694                 itransform = len(transforms) + 1  # 1-based index pointing to the transform
     695                 transforms.append(trs)
     696             pass
     697             log.info(" itransform : %s " % itransform )
     698 
     699             node_planes = node.planes
     700             if len(node_planes) == 0:
     701                 planeIdx = 0
     702                 planeNum = 0
     703             else:
     704                 planeIdx = len(planes) + 1   # 1-based index pointing to the first plane for the node
     705                 planeNum = len(node_planes)
     706                 planes.extend(node_planes)
     707             pass
     708             log.debug("serialize_r idx %3d itransform %2d planeIdx %2d " % (idx, itransform, planeIdx))
     709 
     710             buf[idx] = node.as_array(itransform, planeIdx, planeNum)
     711 
     712             if node.left is not None and node.right is not None:
     713                 serialize_r( node.left,  2*idx+1)
     714                 serialize_r( node.right, 2*idx+2)
     715             pass
     716         pass
     717 
     718         serialize_r(self, 0)
     719 
     720         tbuf = np.vstack(transforms).reshape(-1,4,4) if len(transforms) > 0 else None
     721         pbuf = np.vstack(planes).reshape(-1,4) if len(planes) > 0 else None
     722 
     723         log.debug("serialized CSG of height %2d into buf with %3d nodes, %3d transforms, %3d planes, meta %r " % (self.height, len(buf), len(transforms), len(planes), self.meta ))
     724         assert tbuf is not None
     725 
     726         return buf, tbuf, pbuf
     727 

::

    1020     def as_array(self, itransform=0, planeIdx=0, planeNum=0):
    1021         """
    1022         Both primitive and internal nodes:
    1023 
    1024         * q2.u.w : CSG type code eg CSG_UNION, CSG_DIFFERENCE, CSG_INTERSECTION, CSG_SPHERE, CSG_BOX, ... 
    1025         * q3.u.w : 1-based transform index, 0 for None
    1026 
    1027         Primitive nodes only:
    1028 
    1029         * q0 : 4*float parameters eg center and radius for sphere
    1030 
    1031         """
    1032         arr = np.zeros( (self.NJ, self.NK), dtype=np.float32 )
    ....
    1060         if len(self.planes) > 0:
    1061             assert planeIdx > 0 and planeNum > 3, (planeIdx, planeNum)  # 1-based plane index
    1062             arr.view(np.uint32)[Q0,X] = planeIdx   # cf NNode::planeIdx
    1063             arr.view(np.uint32)[Q0,Y] = planeNum   # cf NNode::planeNum
    1064         pass
    1065 
    1066         arr.view(np.uint32)[Q2,W] = self.typ
    1067 
    1068         return arr








reproduce plane loading issue in X4SolidLoadTest
---------------------------------------------------

::

     11 int main(int argc, char** argv)
     12 {
     13     OPTICKS_LOG(argc, argv);
     14 
     15     int lvIdx = SSys::getenvint("LV",65) ;
     16     std::string csgpath = BFile::FormPath( X4::X4GEN_DIR, BStr::concat("x", BStr::utoa(lvIdx,3, true), NULL)) ;
     17 
     18     LOG(info) << " lvIdx " << lvIdx << " csgpath " << csgpath ;
     19 
     20     NCSGList* ls = NCSGList::Load(csgpath.c_str());
     21     assert(ls);
     22     return 0 ;
     23 }   

::

    epsilon:extg4 blyth$ LV=66 X4SolidLoadTest
    2018-07-29 19:21:59.566 INFO  [5433858] [main@18]  lvIdx 66 csgpath /tmp/blyth/opticks/x4gen/x066
    2018-07-29 19:21:59.567 INFO  [5433858] [NCSGList::load@133] NCSGList::load VERBOSITY 0 basedir /tmp/blyth/opticks/x4gen/x066 txtpath /tmp/blyth/opticks/x4gen/x066/csg.txt nbnd 2
    2018-07-29 19:21:59.568 INFO  [5433858] [NCSGData::loadsrc@310]  loadsrc DONE  ht  2 nn    7 snd 7,4,4 nd NULL str 5,4,4 tr NULL gtr NULL pln 6,4
    Assertion failed: (idx < m_num_planes), function getSrcPlanes, file /Users/blyth/opticks/npy/NCSGData.cpp, line 712.
    Abort trap: 6
    epsilon:extg4 blyth$ 


    epsilon:extg4 blyth$ LV=65 X4SolidLoadTest
    2018-07-29 19:21:54.926 INFO  [5433729] [main@18]  lvIdx 65 csgpath /tmp/blyth/opticks/x4gen/x065
    2018-07-29 19:21:54.928 INFO  [5433729] [NCSGList::load@133] NCSGList::load VERBOSITY 0 basedir /tmp/blyth/opticks/x4gen/x065 txtpath /tmp/blyth/opticks/x4gen/x065/csg.txt nbnd 2
    2018-07-29 19:21:54.929 INFO  [5433729] [NCSGData::loadsrc@310]  loadsrc DONE  ht  2 nn    7 snd 7,4,4 nd NULL str 5,4,4 tr NULL gtr NULL pln 5,4
    2018-07-29 19:21:54.930 ERROR [5433729] [*NCSG::import_r@547] import_r node->gtransform_idx 1
    2018-07-29 19:21:54.930 ERROR [5433729] [*NCSG::import_r@547] import_r node->gtransform_idx 1
    Assertion failed: (idx < m_num_planes), function getSrcPlanes, file /Users/blyth/opticks/npy/NCSGData.cpp, line 712.
    Abort trap: 6

::

    (lldb) p typecode
    (OpticksCSG_t) $0 = CSG_CONVEXPOLYHEDRON
    (lldb) f 5
    frame #5: 0x0000000103ae0972 libNPY.dylib`NCSG::import_srcplanes(this=0x0000000106c00410, node=0x0000000106c02ab0) at NCSG.cpp:702
       699 	    assert( node->planes.size() == 0u );
       700 	
       701 	    std::vector<glm::vec4> _planes ;  
    -> 702 	    m_csgdata->getSrcPlanes(_planes, idx, num_plane ); 
       703 	    assert( _planes.size() == num_plane ) ; 
       704 	
       705 	    cpol->set_planes(_planes);     
    (lldb) p num_plane
    (unsigned int) $1 = 5
    (lldb) p idx
    (unsigned int) $2 = 4294967295
    (lldb) p (int)idx
    (int) $3 = -1
    (lldb) p (int)idx
    (int) $3 = -1
    (lldb) p iplane
    (unsigned int) $4 = 0
    (lldb) p num_plane
    (unsigned int) $5 = 5
    (lldb) 


looks like an off-by-one
----------------------------

definitive truth is the kernel code::

      11 using namespace optix;
      12 
      13 rtBuffer<float4> planBuffer ;
      14 
      15 static __device__
      16 void csg_bounds_convexpolyhedron(const Part& pt, optix::Aabb* aabb, optix::Matrix4x4* tr, const unsigned& planeOffset )
      17 {
      18     const quad& q2 = pt.q2 ;
      19     const quad& q3 = pt.q3 ;
      20 
      21     unsigned planeIdx = pt.planeIdx() ;
      22     unsigned planeNum = pt.planeNum() ;
      23 
      24     rtPrintf("## csg_bounds_convexpolyhedron planeIdx %u planeNum %u planeOffset %u  \n", planeIdx, planeNum, planeOffset );
      25     unsigned planeBase = planeIdx-1+planeOffset ;
      26 

      44 static __device__
      45 bool csg_intersect_convexpolyhedron(const Part& pt, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction, const unsigned& planeOffset )
      46 {
      47     unsigned planeIdx = pt.planeIdx() ;
      48     unsigned planeNum = pt.planeNum() ;
      49     unsigned planeBase = planeIdx-1+planeOffset ;
      50 
      51 
      52 #ifdef CSG_INTERSECT_CONVEXPOLYHEDRON_TEST
      53     const float3& o = ray_origin ;
      54     const float3& d = ray_direction ;
      55     rtPrintf("\n## csg_intersect_convexpolyhedron planeIdx %u planeNum %u planeOffset %u planeBase %u  \n", planeIdx, planeNum, planeOffset, planeBase );
      56     rtPrintf("## csg_intersect_convexpolyhedron o: %10.3f %10.3f %10.3f  d: %10.3f %10.3f %10.3f \n", o.x, o.y, o.z, d.x, d.y, d.z );
      57 #endif
      58 
      59     float t0 = -CUDART_INF_F ;
      60     float t1 =  CUDART_INF_F ;
      61 
      62     float3 t0_normal = make_float3(0.f);
      63     float3 t1_normal = make_float3(0.f);
      64 
      65     //for(unsigned i=0 ; i < planeNum && t0 < t1  ; i++)
      66     for(unsigned i=0 ; i < planeNum ; i++)
      67     {
      68         float4 plane = planBuffer[planeBase+i];
      69         float3 n = make_float3(plane);
      70         float dplane = plane.w ;
      71 

::

     05 struct Part
      6 {
      7 
      8     quad q0 ;
      9     quad q1 ;
     10     quad q2 ;
     11     quad q3 ;
     12 
     13     __device__ unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
     14     __device__ bool        complement() const { return q3.u.w & 0x80000000 ; }
     15 
     16 
     17     __device__ unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
     18     __device__ unsigned planeNum()      const { return q0.u.y ; }
     19 
     20     __device__ void setPlaneIdx(unsigned idx){  q0.u.x = idx ; }
     21     __device__ void setPlaneNum(unsigned num){  q0.u.y = num ; }
     22 
     23 



::

     680 void NCSG::import_srcplanes(nnode* node)
     681 {
     682     assert( node->has_planes() );
     683 
     684     nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
     685     assert(cpol);
     686 
     687     unsigned iplane = node->planeIdx() ;   // 1-based idx ?
     688     unsigned num_plane = node->planeNum() ;
     689     unsigned idx = iplane - 1 ;
     690 
     691     if(m_verbosity > 3)
     692     {
     693     LOG(info) << "NCSG::import_planes"
     694               << " iplane " << iplane
     695               << " num_plane " << num_plane
     696               ;
     697     }
     698 
     699     assert( node->planes.size() == 0u );
     700 
     701     std::vector<glm::vec4> _planes ;
     702     m_csgdata->getSrcPlanes(_planes, idx, num_plane );
     703     assert( _planes.size() == num_plane ) ;
     704 
     705     cpol->set_planes(_planes);
     706     assert( cpol->planes.size() == num_plane );
     707 }


::

    epsilon:npy blyth$ ll /tmp/blyth/opticks/x4gen/x065/1/
    total 40
    drwxr-xr-x  7 blyth  wheel  224 Jul 29 14:41 .
    drwxr-xr-x  6 blyth  wheel  192 Jul 29 17:43 ..
    -rw-r--r--  1 blyth  wheel  528 Jul 29 19:02 srcnodes.npy
    -rw-r--r--  1 blyth  wheel  400 Jul 29 19:02 srctransforms.npy
    -rw-r--r--  1 blyth  wheel  160 Jul 29 19:02 srcplanes.npy
    -rw-r--r--  1 blyth  wheel   96 Jul 29 19:02 srcidx.npy
    -rw-r--r--  1 blyth  wheel   45 Jul 29 19:02 meta.json
    epsilon:npy blyth$ 


::

    epsilon:npy blyth$ np.py /tmp/blyth/opticks/x4gen/x065/
    /tmp/blyth/opticks/x4gen/x065
    /tmp/blyth/opticks/x4gen/x065/csg.txt : 2 
    /tmp/blyth/opticks/x4gen/x065/GItemList/GMaterialLib.txt : 3 
    /tmp/blyth/opticks/x4gen/x065/GItemList/GSurfaceLib.txt : 1 
    /tmp/blyth/opticks/x4gen/x065/0/srctransforms.npy : (1, 4, 4) 
    /tmp/blyth/opticks/x4gen/x065/0/srcnodes.npy : (1, 4, 4) 
    /tmp/blyth/opticks/x4gen/x065/0/srcidx.npy : (1, 4) 

    /tmp/blyth/opticks/x4gen/x065/1/srctransforms.npy : (5, 4, 4) 
    /tmp/blyth/opticks/x4gen/x065/1/srcnodes.npy : (7, 4, 4) 
    /tmp/blyth/opticks/x4gen/x065/1/srcplanes.npy : (5, 4) 
    /tmp/blyth/opticks/x4gen/x065/1/srcidx.npy : (1, 4) 
    epsilon:npy blyth$ 




