g4Live_gltf_shakedown
========================


DYB Nodes for glTF check viz
--------------------------------

Debug by editing the glTF to pick particular nodes::

    329578   "scenes": [
    329579     {
    329580       "nodes": [
    329581         3199
    329582       ]
    329583     }

::

   3199 : single pmt (with frame false looks correct, with frame true mangled)
   3155 : AD  (view starts from above the lid) (with frame false PMT all pointing in one direction, with frame true correct)
   3147 : pool with 2 ADs etc..


NEXT
-----

* comparison of GGeo instances from two routes 

  * easiest way is to persist and compare files 

* need to get thru to raytracing the direct geometry 



Three Solids X4Mesh skipped still 
------------------------------------

::

    443      std::vector<unsigned> skips = {27, 29, 33 };
    444 
    445      if(mh->csgnode == NULL)
    446      {
    447          mh->csgnode = X4Solid::Convert(solid) ;  // soIdx 33 giving analytic problems too 
    448 
    449          bool placeholder = std::find( skips.begin(), skips.end(), nd->soIdx ) != skips.end()  ;
    450 
    451          mh->mesh = placeholder ? X4Mesh::Placeholder(solid) : X4Mesh::Convert(solid) ;
    452 



Comparing geocache : some large differences in groupvel ? UNDERSTOOD
------------------------------------------------------------------------

Huh : the old geocache material groupvel always 300, but the 
new one is varying.  Was that a postcache fixup ? 

* Ah-ha : the fixup was done postcache (GMaterialLib::postLoadFromCache) 
  SO THE 300. IN THE OLD GEOCACHE ARE UNDERSTOOD : DIFFERENCE IS UNDERSTOOD 


::

    055 void GMaterialLib::postLoadFromCache()
     56 {
     ..
     69     bool groupvel = !m_ok->hasOpt("nogroupvel") ;
     70 

    119     if(groupvel)   // unlike the other material changes : this one is ON by default, so long at not swiched off with --nogroupvel
    120     {
    121        bool debug = false ;
    122        replaceGROUPVEL(debug);
    123     }
    124 




::

    In [58]: cat geocache.py 
    #!/usr/bin/env python

    import os, numpy as np

    idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )
    idp2_ = lambda _:os.path.expandvars("$IDPATH2/%s" % _ )


    if __name__ == '__main__':
        aa = np.load(idp_("GMaterialLib/GMaterialLib.npy"))
        bb = np.load(idp2_("GMaterialLib/GMaterialLib.npy"))
        assert aa.shape == bb.shape
        print aa.shape

        for i in range(len(aa)):
            a = aa[i]  
            b = bb[i]  
            assert len(a) == 2 
            assert len(b) == 2 

            g0 = a[0] - b[0] 
            g1 = a[1] - b[1] 

            assert g0.shape == g1.shape

            print i, g0.shape, "g0max: ", np.max(g0), "g1max: ", np.max(g1)




::

    In [51]: aa[:,1,:,0]
    Out[51]: 
    array([[300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.],
           ...,
           [300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.]], dtype=float32)

    In [52]: aa[:,1,:,0].shape
    Out[52]: (38, 39)

    In [53]: aa[:,1,:,0].min()
    Out[53]: 300.0

    In [54]: aa[:,1,:,0].max()
    Out[54]: 300.0

    In [55]: bb[:,1,:,0]
    Out[55]: 
    array([[206.2414, 206.2414, 206.2414, ..., 200.9359, 201.9052, 202.8228],
           [206.2414, 206.2414, 206.2414, ..., 200.9359, 201.9052, 202.8228],
           [205.0564, 205.0564, 205.0564, ..., 199.8321, 200.6891, 201.5005],
           ...,
           [299.7924, 299.7924, 299.7924, ..., 299.7924, 299.7924, 299.7924],
           [299.7924, 299.7924, 299.7924, ..., 299.7924, 299.7924, 299.7924],
           [300.    , 300.    , 300.    , ..., 300.    , 300.    , 300.    ]], dtype=float32)

    In [56]: bb[:,1,:,0].min()
    Out[56]: 118.98735

    In [57]: bb[:,1,:,0].max()
    Out[57]: 300.0




::

    In [22]: run geocache.py 
    (38, 2, 39, 4)
    0 (39, 4) g0max:  0.015625 g1max:  181.01265
    1 (39, 4) g0max:  0.015625 g1max:  181.01265
    2 (39, 4) g0max:  0.015625 g1max:  180.42665
    3 (39, 4) g0max:  0.015625 g1max:  178.10599
    4 (39, 4) g0max:  0.00024414062 g1max:  94.38103
    5 (39, 4) g0max:  0.005859375 g1max:  93.02899
    6 (39, 4) g0max:  0.005859375 g1max:  93.02899
    7 (39, 4) g0max:  0.005859375 g1max:  93.02899
    8 (39, 4) g0max:  0.005859375 g1max:  93.02899
    9 (39, 4) g0max:  0.0 g1max:  0.20755005
    10 (39, 4) g0max:  0.0 g1max:  0.20755005
    11 (39, 4) g0max:  0.0 g1max:  0.20755005
    12 (39, 4) g0max:  0.0 g1max:  0.20755005
    13 (39, 4) g0max:  0.00024414062 g1max:  94.38103
    14 (39, 4) g0max:  0.0 g1max:  0.28848267
    15 (39, 4) g0max:  0.0 g1max:  0.0
    16 (39, 4) g0max:  0.0 g1max:  0.20755005
    17 (39, 4) g0max:  0.0 g1max:  0.20755005
    18 (39, 4) g0max:  0.0 g1max:  0.20755005
    19 (39, 4) g0max:  0.0 g1max:  0.20755005
    20 (39, 4) g0max:  0.0 g1max:  0.20755005
    21 (39, 4) g0max:  0.0 g1max:  0.31243896
    22 (39, 4) g0max:  0.0 g1max:  0.20755005
    23 (39, 4) g0max:  0.0 g1max:  0.20755005
    24 (39, 4) g0max:  0.0 g1max:  0.20755005
    25 (39, 4) g0max:  0.0 g1max:  0.20755005
    26 (39, 4) g0max:  0.0 g1max:  0.20755005
    27 (39, 4) g0max:  0.0 g1max:  0.20755005
    28 (39, 4) g0max:  0.015625 g1max:  180.42665
    29 (39, 4) g0max:  0.0 g1max:  0.20755005
    30 (39, 4) g0max:  0.0 g1max:  0.20755005
    31 (39, 4) g0max:  0.0 g1max:  0.20755005
    32 (39, 4) g0max:  0.0 g1max:  0.20755005
    33 (39, 4) g0max:  0.0 g1max:  0.20755005
    34 (39, 4) g0max:  0.0 g1max:  0.20755005
    35 (39, 4) g0max:  0.0 g1max:  0.20755005
    36 (39, 4) g0max:  0.0 g1max:  0.20755005
    37 (39, 4) g0max:  0.0 g1max:  0.0




FIXED : Comparing geocache : material lib ordering and test materials
---------------------------------------------------------------------------

* sort material order

  * sorting done by GPropertyLib::close, based on Order from m_attrnames 

::

    338 std::map<std::string, unsigned int>& GPropertyLib::getOrder()
    339 {
    340     return m_attrnames->getOrder() ;
    341 }


GPropertyLib::init loads the prefs including the order::

    318     m_attrnames = new OpticksAttrSeq(m_ok, m_type);
    319     m_attrnames->loadPrefs(); // color.json, abbrev.json and order.json 
    320     LOG(debug) << "GPropertyLib::init loadPrefs-DONE " ;

::

    OpticksResourceTest:

                     detector_base :  Y :      /usr/local/opticks/opticksdata/export/DayaBay


    epsilon:issues blyth$ ll /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/
    -rw-r--r--  1 blyth  staff  612 Apr  4 14:26 abbrev.json
    -rw-r--r--  1 blyth  staff  660 Apr  4 14:26 color.json
    -rw-r--r--  1 blyth  staff  795 Apr  4 14:26 order.json


::

   OPTICKS_KEY=CX4GDMLTest.X4PhysicalVolume.World0xc15cfc0_PV.828722902b5e94dab05ac248329ffebe OpticksResourceTest 


Kludge symbolic link to try to access the prefs with the g4live running::

    epsilon:~ blyth$ cd /usr/local/opticks-cmake-overhaul/opticksdata/export/
    epsilon:export blyth$ ln -s DayaBay CX4GDMLTest


* add test materials

::

    export IDPATH2=/usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1

    epsilon:ana blyth$ python geocache.py 
    (38, 2, 39, 4)
    (36, 2, 39, 4)

::

    epsilon:1 blyth$ head -5 $IDPATH/GItemList/GMaterialLib.txt 
    GdDopedLS
    LiquidScintillator
    Acrylic
    MineralOil
    Bialkali
    epsilon:1 blyth$ head -5 $IDPATH2/GItemList/GMaterialLib.txt 
    PPE
    MixGas
    Air
    Bakelite
    Foam




FIXED : material names with slashes mess up boundary spec 
------------------------------------------------------------

* fixed using basenames

cfg4-;cfg4-c;om-;TEST=CX4GDMLTest om-d::

    2018-06-23 16:30:36.316 INFO  [25301620] [GParts::close@802] GParts::close START  verbosity 0
    2018-06-23 16:30:36.316 FATAL [25301620] [GBnd::init@27] GBnd::init bad boundary spec, expecting 4 elements spec /dd/Materials/Vacuum////dd/Materials/Vacuum nelem 10
    Assertion failed: (nelem == 4), function init, file /Users/blyth/opticks-cmake-overhaul/ggeo/GBnd.cc, line 34.
    Process 19616 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (CX4GDMLTest) stopped.
    (lldb) 




FIXED : Slow convert due to CSG node nudger running at node(not mesh) level ?
-------------------------------------------------------------------------------- 

* moving the nudging to mesh level, gives drastic speedup : now DYB near
  conversion from G4 model to Opticks GGeo and writes out glTF in 5 seconds.

* looks like the slow convert, was related to not having the displacements 
  done already, nevertheless : if this processing can be moved to mesh level 
  ot should be 



X4PhysicalVolume::convertNode::

    434 
    435      Mh* mh = m_sc->get_mesh_for_node( ndIdx );  // node->mesh via soIdx (the local mesh index)
    436 
    437      std::vector<unsigned> skips = {27, 29, 33 };
    438 
    439      if(mh->csg == NULL)
    440      {
    441          //convertSolid(mh, solid);
    442          mh->csg = X4Solid::Convert(solid) ;  // soIdx 33 giving analytic problems too 
    443 
    444          bool placeholder = std::find( skips.begin(), skips.end(), nd->soIdx ) != skips.end()  ;
    445 
    446          mh->mesh = placeholder ? X4Mesh::Placeholder(solid) : X4Mesh::Convert(solid) ;
    447 
    448          mh->vtx = mh->mesh->m_x4src_vtx ;
    449          mh->idx = mh->mesh->m_x4src_idx ;
    450      }
    451 
    452      assert( mh->csg );
    453 
    454      // can this be done at mesh level (ie within the above bracket) ?
    455      // ... would be a big time saving 
    456      // ... see how the boundary is used, also check GParts 
    457 
    458      mh->csg->set_boundary( boundaryName.c_str() ) ;
    459 
    460      NCSG* csg = NCSG::FromNode( mh->csg, NULL );
    461      assert( csg ) ;
    462      assert( csg->isUsedGlobally() );
    463 
    464      const GMesh* mesh = mh->mesh ;   // hmm AssimpGGeo::convertMeshes does deduping/fixing before inclusion in GVolume(GNode) 
    465 
    466      GParts* pts = GParts::make( csg, boundaryName.c_str(), m_verbosity  );  // see GScene::createVolume 
    467 


* WHY does NCSG require nnode to have boundary spec char* ? 

  * Suspect nnode does not need boundary any more ?
  * hmm actually that was probably a convenience for tboolean- passing boundaries in from python,
    so need to keep the capability
  * GParts really needs this spec, as it has a GBndLib to convert the spec 
    into a bndIdx for laying down in buffers


* guess that GParts needs to be at node level, peer with GVolume 






DONE : initial implementation to convert G4DisplacedSolid into nnode CSG 
---------------------------------------------------------------------------

::

     87 G4BooleanSolid::G4BooleanSolid( const G4String& pName,
     88                                       G4VSolid* pSolidA ,
     89                                       G4VSolid* pSolidB ,
     90                                 const G4Transform3D& transform    ) :
     91   G4VSolid(pName), fAreaRatio(0.), fStatistics(1000000), fCubVolEpsilon(0.001),
     92   fAreaAccuracy(-1.), fCubicVolume(0.), fSurfaceArea(0.),
     93   fRebuildPolyhedron(false), fpPolyhedron(0), createdDisplacedSolid(true)
     94 {
     95   fPtrSolidA = pSolidA ;
     96   fPtrSolidB = new G4DisplacedSolid("placedB",pSolidB,transform) ;
     97 }

::

     70 G4DisplacedSolid::G4DisplacedSolid( const G4String& pName,
     71                                           G4VSolid* pSolid ,
     72                                     const G4Transform3D& transform  )
     73   : G4VSolid(pName), fRebuildPolyhedron(false), fpPolyhedron(0)
     74 {
     75   fPtrSolid = pSolid ;
     76   fDirectTransform = new G4AffineTransform(transform.getRotation().inverse(),
     77                                            transform.getTranslation()) ;
     78 
     79   fPtrTransform    = new G4AffineTransform(transform.getRotation().inverse(),
     80                                            transform.getTranslation()) ;
     81   fPtrTransform->Invert() ;
     82 }


g4-gcd::

     152 void G4GDMLWriteSolids::
     153 BooleanWrite(xercesc::DOMElement* solElement,
     154              const G4BooleanSolid* const boolean)
     155 {
     156    G4int displaced=0;
     157 
     158    G4String tag("undefined");
     159    if (dynamic_cast<const G4IntersectionSolid*>(boolean))
     160      { tag = "intersection"; } else
     161    if (dynamic_cast<const G4SubtractionSolid*>(boolean))
     162      { tag = "subtraction"; } else
     163    if (dynamic_cast<const G4UnionSolid*>(boolean))
     164      { tag = "union"; }
     165 
     166    G4VSolid* firstPtr = const_cast<G4VSolid*>(boolean->GetConstituentSolid(0));
     167    G4VSolid* secondPtr = const_cast<G4VSolid*>(boolean->GetConstituentSolid(1));
     168 
     169    G4ThreeVector firstpos,firstrot,pos,rot;
     170 
     171    // Solve possible displacement of referenced solids!
     172    //
     173    while (true)
     174    {
     175       if ( displaced>8 )
     ///                 ... error message ...
     ...
     186       if (G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(firstPtr))
     187       {
     188          firstpos += disp->GetObjectTranslation();
     189          firstrot += GetAngles(disp->GetObjectRotation());
     ///
     ///      adding angles ... hmm looks fishy 
     ///
     190          firstPtr = disp->GetConstituentMovedSolid();
     191          displaced++;
     ///
     ///   can understand why you might have one displacement ?
     ///   but how you manage to have 8 displacements ? 
     ///
     192          continue;
     193       }
     194       break;
     195    }
     196    displaced = 0;

     ...
     221    AddSolid(firstPtr);   // At first add the constituent solids!
     222    AddSolid(secondPtr);
     223 
     224    const G4String& name = GenerateName(boolean->GetName(),boolean);
     225    const G4String& firstref = GenerateName(firstPtr->GetName(),firstPtr);
     226    const G4String& secondref = GenerateName(secondPtr->GetName(),secondPtr);
     227 
     228    xercesc::DOMElement* booleanElement = NewElement(tag);
     229    booleanElement->setAttributeNode(NewAttribute("name",name));
     230    xercesc::DOMElement* firstElement = NewElement("first");
     231    firstElement->setAttributeNode(NewAttribute("ref",firstref));
     232    booleanElement->appendChild(firstElement);
     233    xercesc::DOMElement* secondElement = NewElement("second");
     234    secondElement->setAttributeNode(NewAttribute("ref",secondref));
     235    booleanElement->appendChild(secondElement);
     236    solElement->appendChild(booleanElement);
     237      // Add the boolean solid AFTER the constituent solids!
     238 
     239    if ( (std::fabs(pos.x()) > kLinearPrecision)
     240      || (std::fabs(pos.y()) > kLinearPrecision)
     241      || (std::fabs(pos.z()) > kLinearPrecision) )
     242    {
     243      PositionWrite(booleanElement,name+"_pos",pos);
     244    }
     245 
     246    if ( (std::fabs(rot.x()) > kAngularPrecision)
     247      || (std::fabs(rot.y()) > kAngularPrecision)
     248      || (std::fabs(rot.z()) > kAngularPrecision) )
     249    {
     250      RotationWrite(booleanElement,name+"_rot",rot);
     251    }
     252 
     253    if ( (std::fabs(firstpos.x()) > kLinearPrecision)
     254      || (std::fabs(firstpos.y()) > kLinearPrecision)
     255      || (std::fabs(firstpos.z()) > kLinearPrecision) )
     256    {
     257      FirstpositionWrite(booleanElement,name+"_fpos",firstpos);
     258    }
     259 
     260    if ( (std::fabs(firstrot.x()) > kAngularPrecision)
     261      || (std::fabs(firstrot.y()) > kAngularPrecision)
     262      || (std::fabs(firstrot.z()) > kAngularPrecision) )
     263    {
     264      FirstrotationWrite(booleanElement,name+"_frot",firstrot);
     265    }
     266 }


::

     .80 void G4GDMLReadSolids::
      81 BooleanRead(const xercesc::DOMElement* const booleanElement, const BooleanOp op)
      82 {
     ...
     154    G4VSolid* firstSolid = GetSolid(GenerateName(first));
     155    G4VSolid* secondSolid = GetSolid(GenerateName(scnd));
     156 
     157    G4Transform3D transform(GetRotationMatrix(rotation),position);
     158 
     159    if (( (firstrotation.x()!=0.0) || (firstrotation.y()!=0.0)
     160                                   || (firstrotation.z()!=0.0))
     161     || ( (firstposition.x()!=0.0) || (firstposition.y()!=0.0)
     162                                   || (firstposition.z()!=0.0)))
     163    {
     164       G4Transform3D firsttransform(GetRotationMatrix(firstrotation),
     165                                    firstposition);
     166       firstSolid = new G4DisplacedSolid(GenerateName("displaced_"+first),
     167                                         firstSolid, firsttransform);
     168    }
     169 
     170    if (op==UNION)
     171      { new G4UnionSolid(name,firstSolid,secondSolid,transform); } else
     172    if (op==SUBTRACTION)
     173      { new G4SubtractionSolid(name,firstSolid,secondSolid,transform); } else
     174    if (op==INTERSECTION)
     175      { new G4IntersectionSolid(name,firstSolid,secondSolid,transform); }
     176 }

::

    132 G4RotationMatrix
    133 G4GDMLReadDefine::GetRotationMatrix(const G4ThreeVector& angles)
    134 {
    135    G4RotationMatrix rot;
    136 
    137    rot.rotateX(angles.x());
    138    rot.rotateY(angles.y());
    139    rot.rotateZ(angles.z());
    140    rot.rectify();  // Rectify matrix from possible roundoff errors
    141 
    142    return rot;




G4GDMLWriteDefine.hh::

     58     void RotationWrite(xercesc::DOMElement* element,
     59                     const G4String& name, const G4ThreeVector& rot)
     60          { Rotation_vectorWrite(element,"rotation",name,rot); }
     61     void PositionWrite(xercesc::DOMElement* element,
     62                     const G4String& name, const G4ThreeVector& pos)
     63          { Position_vectorWrite(element,"position",name,pos); }
     64     void FirstrotationWrite(xercesc::DOMElement* element,
     65                     const G4String& name, const G4ThreeVector& rot)
     66          { Rotation_vectorWrite(element,"firstrotation",name,rot); }
     67     void FirstpositionWrite(xercesc::DOMElement* element,
     68                     const G4String& name, const G4ThreeVector& pos)
     69          { Position_vectorWrite(element,"firstposition",name,pos); }
     70     void AddPosition(const G4String& name, const G4ThreeVector& pos)
     71          { Position_vectorWrite(defineElement,"position",name,pos


gdml.py::

     * no handling of : firstposition, firstrotation


     166 class Boolean(Geometry):
     167     firstref = property(lambda self:self.elem.find("first").attrib["ref"])
     168     secondref = property(lambda self:self.elem.find("second").attrib["ref"])
     169 
     170     position = property(lambda self:self.find1_("position"))
     171     rotation = property(lambda self:self.find1_("rotation"))
     172     scale = None
     173     secondtransform = property(lambda self:construct_transform(self))
     174 
     175     first = property(lambda self:self.g.solids[self.firstref])
     176     second = property(lambda self:self.g.solids[self.secondref])
     177 
     ...
     183     def as_ncsg(self):
     ...
     188         left = self.first.as_ncsg()
     189         right = self.second.as_ncsg()
     ...
     194         right.transform = self.secondtransform
     195 
     196         cn = CSG(self.operation, name=self.name)
     197         cn.left = left
     198         cn.right = right
     199         return cn


::

      31 def construct_transform(obj):
      32     tla = obj.position.xyz if obj.position is not None else None
      33     rot = obj.rotation.xyz if obj.rotation is not None else None
      34     sca = obj.scale.xyz if obj.scale is not None else None
      35     order = "trs"
      36 
      37     #elem = filter(None, [tla,rot,sca])
      38     #if len(elem) > 1:
      39     #    log.warning("construct_transform multi %s " % repr(obj))
      40     #pass
      41 
      42     return make_transform( order, tla, rot, sca , three_axis_rotate=True, transpose_rotation=True, suppress_identity=False, dtype=np.float32 )
      43 


::

    258 def make_transform( order, tla, rot, sca, dtype=np.float32, suppress_identity=True, three_axis_rotate=False, transpose_rotation=False):
    259     """
    260     :param order: string containing "s" "r" and "t", standard order is "trs" meaning t*r*s  ie scale first, then rotate, then translate 
    261     :param tla: tx,ty,tz tranlation dists eg 0,0,0 for no translation 
    262     :param rot: ax,ay,az,angle_degrees  eg 0,0,1,45 for 45 degrees about z-axis
    263     :param sca: sx,sy,sz eg 1,1,1 for no scaling 
    264     :return mat: 4x4 numpy array 
    265 
    266     All arguments can be specified as comma delimited string, list or numpy array
    267 
    268     Translation of npy/tests/NGLMTest.cc:make_mat
    269     """
    270 
    271     if tla is None and rot is None and sca is None and suppress_identity:
    272         return None
    273 
    274     identity = np.eye(4, dtype=dtype)
    275     m = np.eye(4, dtype=dtype)
    276     for c in order:
    277         if c == 's':
    278             m = make_scale(sca, m)
    279         elif c == 'r':
    280             if three_axis_rotate:
    281                 m = rotate_three_axis(rot, m, transpose=transpose_rotation )
    282             else:
    283                 m = rotate(rot, m, transpose=transpose_rotation )
    284             pass
    285         elif c == 't':
    286             m = translate(tla, m)
    287         else:
    288             assert 0
    289         pass
    290     pass
    291 
    292     if suppress_identity and np.all( m == identity ):
    293         #log.warning("supressing identity transform")
    294         return None
    295     pass
    296     return m




FIXED : glTF viz shows messed up transforms
----------------------------------------------

Debug by editing the glTF to pick particular nodes::

    329578   "scenes": [
    329579     {
    329580       "nodes": [
    329581         3199
    329582       ]
    329583     }


::

   3199 : single pmt (with frame false looks correct, with frame true mangled)
   3155 : AD  (view starts from above the lid) (with frame false PMT all pointing in one direction, with frame true correct)
   3147 : pool with 2 ADs etc..


Similar trouble before
~~~~~~~~~~~~~~~~~~~~~~~~~

Every time, gets troubles from transforms...

* :doc:`gdml_gltf_transforms`


Debugging Approach ?
~~~~~~~~~~~~~~~~~~~~~~~

* compare the GGeo transforms from the two streams 
* simplify transform handling : avoid multiple holdings of transforms, 
  
Observations

* assembly of the PMT within its "frame" (of 5 parts) only involves 
  translation in z : so getting that correct could be deceptive as no rotation   


Switching to frame gets PMT pointing correct, but seems mangled inside themselves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* mangled : the base poking thru the front 


::

     20 glm::mat4* X4Transform3D::GetLocalTransform(const G4VPhysicalVolume* const pv, bool frame)
     21 {    
     22     glm::mat4* transform = NULL ;
     23     if(frame)
     24     {
     25         const G4RotationMatrix* rotp = pv->GetFrameRotation() ;
     26         G4ThreeVector    tla = pv->GetFrameTranslation() ;
     27         G4Transform3D    tra(rotp ? *rotp : G4RotationMatrix(),tla);
     28         transform = new glm::mat4(Convert( tra ));
     29     }   
     30     else
     31     {
     32         G4RotationMatrix rot = pv->GetObjectRotationValue() ;  // obj relative to mother
     33         G4ThreeVector    tla = pv->GetObjectTranslation() ; 
     34         G4Transform3D    tra(rot,tla);
     35         transform = new glm::mat4(Convert( tra ));
     36     }   
     37     return transform ;
     38 }   




FIXED : bad mesh association, missing meshes
------------------------------------------------

Also add metadata extras to allow to navigate the gltf.  Suspect 
are getting bad mesh association, as unexpected lots of repeated mesh.

Huh : only 35 meshes, (expect ~250) but the expected 12k nodes.

Suspect the lvIdx mesh identity.




