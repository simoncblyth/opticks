Full Analytic Route Needs Material Hookup
============================================

As GDML does not contain optical props will need 
to interface to G4DAE material props.  

* better to think of this as geometry merging between the two routes


TODO : C++ alignment  + cross accessors 
--------------------------------------------
 
* need partial alignment check
* cross accessors : to get to corresponding objects in "other" world, just need the right index

* verify oxrap can continue unchanged

* COMPLICATION : tri when operating from cache, does not have the volume tree persisted/saved
  only the merged mesh and its solid related buffers are persisted ?

* Hmm but due to instancing splits of the GMergedMesh this aint so easy ?
  Unless the global one retains all solid info ?


Yep mm0 is holding full solid info::

    In [1]: run mergedmesh.py

    [2017-06-21 12:59:49,242] p1196 {/Users/blyth/opticks/ana/mergedmesh.py:51} INFO - ready buffers from /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 
    -rw-r--r--  1 blyth  staff  96 Jun 14 16:21 /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0/aiidentity.npy
    -rw-r--r--  1 blyth  staff  293600 Jun 14 16:21 /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0/bbox.npy
    ...
    In [2]: mm
    Out[2]: 
               aiidentity : (1, 1, 4) 
                     bbox : (12230, 6) 
               boundaries : (434816, 1) 
            center_extent : (12230, 4) 
                   colors : (225200, 3) 
                 identity : (12230, 4) 
                iidentity : (12230, 4) 
                  indices : (1304448, 1) 
              itransforms : (1, 4, 4) 
                   meshes : (12230, 1) 
                 nodeinfo : (12230, 4) 
                    nodes : (434816, 1) 
                  normals : (225200, 3) 
                  sensors : (434816, 1) 
               transforms : (12230, 16) 
                 vertices : (225200, 3) 


Looks like key method is::


     GMergedMesh::mergeSolidIdentity( GSolid* solid, bool selected ) 

     m_nodeinfo
     m_identity

     394 guint4* GMesh::getNodeInfo()
     395 {
     396     return m_nodeinfo ;
     397 }
     398 guint4 GMesh::getNodeInfo(unsigned int index)
     399 {
     400     return m_nodeinfo[index] ;
     401 }
     402 
     403 guint4* GMesh::getIdentity()
     404 {
     405     return m_identity ;
     406 }
     407 guint4 GMesh::getIdentity(unsigned int index)
     408 {
     409     return m_identity[index] ;
     410 }





::

    2017-06-21 12:33:52.824 INFO  [8897976] [GScene::compareTrees@84] GScene::compareTrees
     ana GNodeLib targetnode 3153 numPV 1660 numLV 1660 numSolids 1660 PV(0) /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 LV(0) /dd/Geometry/AD/lvADE0xc2a78c0
     tri GNodeLib targetnode 3153 numPV 1660 numLV 1660 numSolids 0 PV(0) /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 LV(0) /dd/Geometry/AD/lvADE0xc2a78c0
    Assertion failed: (0 && "GScene::init early exit for gltf==44"), function init, file /Users/blyth/opticks/ggeo/GScene.cc, line 72.
    Process 33509 stopped




DONE : gltftarget(config) and targetnode(metadata)
-----------------------------------------------------

Partial targetnode now available via these routes.



DONE : analytic/triangulated tree alignment for full traversals
------------------------------------------------------------------

Verified same node counts, traversal order and pv/lv identifiers 
in ana/nodelib.py 

Implemented using 2 GGeoLib instances for the GMergedMesh
and 2 GNodeLib instances for the GSolid:

* triangulated directly in GGeo
* analytic within GGeo/GScene


GGeo/GGeoLib
---------------

Normally loaded from cache::


     610 void GGeo::loadFromCache()
     611 {  
     612     LOG(trace) << "GGeo::loadFromCache START" ;
     613 
     614     m_geolib = GGeoLib::load(m_ok);
     615    


GDML File LV volume elements have material refs
--------------------------------------------------

/tmp/g4_00.gdml::

     .121     <material name="/dd/Materials/GdDopedLS0xc2a8ed0" state="solid">
      122       <P unit="pascal" value="101324.946686941"/>
      123       <D unit="g/cm3" value="0.86019954739804"/>
     ....
     3809     <volume name="/dd/Geometry/AD/lvGDS0xbf6cbb8">
     3810       <materialref ref="/dd/Materials/GdDopedLS0xc2a8ed0"/>
     3811       <solidref ref="gds0xc28d3f0"/>
     3812     </volume>
     3813     <volume name="/dd/Geometry/AdDetails/lvOcrGdsInIav0xbf6dd58">
     3814       <materialref ref="/dd/Materials/GdDopedLS0xc2a8ed0"/>
     3815       <solidref ref="OcrGdsInIav0xc405b10"/>
     3816     </volume>
     3817     <volume name="/dd/Geometry/AD/lvIAV0xc404ee8">
     3818       <materialref ref="/dd/Materials/Acrylic0xc02ab98"/>
     3819       <solidref ref="iav0xc346f90"/>
     3820       <physvol name="/dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00">
     3821         <volumeref ref="/dd/Geometry/AD/lvGDS0xbf6cbb8"/>
     3822         <position name="/dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00_pos" unit="mm" x="0" y="0" z="7.5"/>
     3823       </physvol>
     3824       <physvol name="/dd/Geometry/AD/lvIAV#pvOcrGdsInIAV0xbf6b0e0">
     3825         <volumeref ref="/dd/Geometry/AdDetails/lvOcrGdsInIav0xbf6dd58"/>
     3826         <position name="/dd/Geometry/AD/lvIAV#pvOcrGdsInIAV0xbf6b0e0_pos" unit="mm" x="0" y="0" z="1587.21981588594"/>
     3827       </physvol>
     3828     </volume>


analytic/gdml.py::

     861 class Volume(G):
     862     """
     863     ::
     864 
     865         In [15]: for v in gdml.volumes.values():print v.material.shortname
     866         PPE
     867         MixGas
     868         Air
     869         Bakelite
     870         Air
     871         Bakelite
     872         Foam
     873         Aluminium
     874         Air
     875         ...
     876 
     877     """
     878     materialref = property(lambda self:self.elem.find("materialref").attrib["ref"])
     879     solidref = property(lambda self:self.elem.find("solidref").attrib["ref"])
     880     solid = property(lambda self:self.g.solids[self.solidref])
     881     material = property(lambda self:self.g.materials[self.materialref])
     882 


Whats needed for analytic material ?
---------------------------------------

* need boundary "omat/osur/isur/imat" spec strings for all volumes...


In tboolean testing these boundary spec are set manually on the 
csg object of the solids.

    343 container = CSG("box")
    344 container.boundary = args.container

ana/base.py::

    305     container = kwa.get("container","Rock//perfectAbsorbSurface/Vacuum")
    306     testobject = kwa.get("testobject","Vacuum///GlassSchottF2" )


npy/NCSG.cpp sets boundary strings on the NCSG tree instances::

     885 int NCSG::Deserialize(const char* basedir, std::vector<NCSG*>& trees, int verbosity )
     886 {
     ...
     898     NTxt bnd(txtpath.c_str());
     899     bnd.read();
     900     //bnd.dump("NCSG::Deserialize");    
     901 
     902     unsigned nbnd = bnd.getNumLines();
     903 
     904     LOG(info) << "NCSG::Deserialize"
     905               << " VERBOSITY " << verbosity
     906               << " basedir " << basedir
     907               << " txtpath " << txtpath
     908               << " nbnd " << nbnd
     909               ;
     ...
     917     for(unsigned j=0 ; j < nbnd ; j++)
     918     {
     919         unsigned i = nbnd - 1 - j ;
     920         std::string treedir = BFile::FormPath(basedir, BStr::itoa(i));
     921 
     922         NCSG* tree = new NCSG(treedir.c_str());
     923         tree->setIndex(i);
     924         tree->setVerbosity( verbosity );
     925         tree->setBoundary( bnd.getLine(i) );
     926 

Which are serialized from python source via a csg.txt bnd file::

    simon:tboolean-disc-- blyth$ pwd
    /tmp/blyth/opticks/tboolean-disc--
    simon:tboolean-disc-- blyth$ cat csg.txt 
    Rock//perfectAbsorbSurface/Vacuum
    Vacuum///GlassSchottF2


The above is the python CSG testing route, what about full analytic GDML/GLTF  route ? tgltf-gdml

* the boundary from the node/extras of the GLTF is applied to the structural nd in  NScene::import_r

::

    278 nd* NScene::import_r(int idx,  nd* parent, int depth)
    279 {
    280     ygltf::node_t* ynode = getNode(idx);
    281     auto extras = ynode->extras ;
    282     std::string boundary = extras["boundary"] ;
    283 
    284     nd* n = new nd ;   // NB these are structural nodes, not CSG tree nodes
    285 
    286     n->idx = idx ;
    287     n->repeatIdx = 0 ;
    288     n->mesh = ynode->mesh ;
    289     n->parent = parent ;
    290     n->depth = depth ;
    291     n->boundary = boundary ;
    292     n->transform = new nmat4triple( ynode->matrix.data() );
    293     n->gtransform = nd::make_global_transform(n) ;
    294 
    295     for(int child : ynode->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call
    296 
    297     m_nd[idx] = n ;
    298 
    299     return n ;
    300 }




::

    113 tgltf-gdml(){  TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $* ; }

    115 tgltf-gdml--(){ cat << EOP
    116 
    117 import os, logging, sys, numpy as np
    118 
    119 log = logging.getLogger(__name__)
    120 
    121 from opticks.ana.base import opticks_main
    122 from opticks.analytic.treebase import Tree
    123 from opticks.analytic.gdml import GDML
    124 from opticks.analytic.sc import Sc
    125 
    126 args = opticks_main()
    127 
    128 oil = "/dd/Geometry/AD/lvOIL0xbf5e0b8"
    129 #sel = oil
    130 #sel = 3153
    131 sel = 1
    132 idx = 0 
    133 
    134 wgg = GDML.parse()
    135 tree = Tree(wgg.world)
    136 
    137 target = tree.findnode(sel=sel, idx=idx)
    138 
    139 sc = Sc(maxcsgheight=3)
    140 sc.extras["verbosity"] = 1
    141 
    142 tg = sc.add_tree_gdml( target, maxdepth=0)
    143 
    144 path = "$TMP/tgltf/$FUNCNAME.gltf"
    145 gltf = sc.save(path)
    146 
    147 print path      ## <-- WARNING COMMUNICATION PRINT
    148 
    149 EOP
    150 }


    039 tgltf--(){
     40 
     41     tgltf-
     42 
     43     local cmdline=$*
     44     local tgltfpath=${TGLTFPATH:-$TMP/nd/scene.gltf}
     45 
     46     local gltf=1
     47     #local gltf=4  # early exit from GGeo::loadFromGLTF
     48 
     49     op.sh  \
     50             $cmdline \
     51             --debugger \
     52             --gltf $gltf \
     53             --gltfbase $(dirname $tgltfpath) \
     54             --gltfname $(basename $tgltfpath) \
     55             --target 3 \
     56             --animtimemax 10 \
     57             --timemax 10 \
     58             --geocenter \
     59             --eye 1,0,0 \
     60             --dbganalytic \
     61             --tag $(tgltf-tag) --cat $(tgltf-det) \
     62             --save
     63 }



::


    simon:issues blyth$ tgltf-;tgltf-gdml-
    args: 
    [2017-06-20 14:02:53,885] p85498 {/Users/blyth/opticks/analytic/gdml.py:987} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-06-20 14:02:53,923] p85498 {/Users/blyth/opticks/analytic/gdml.py:1001} INFO - wrapping gdml element  
    [2017-06-20 14:02:54,765] p85498 {/Users/blyth/opticks/analytic/sc.py:279} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0
    ...
    [2017-06-20 14:02:57,976] p85498 {/Users/blyth/opticks/analytic/sc.py:304} INFO - saving to /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf 
    [2017-06-20 14:02:58,221] p85498 {/Users/blyth/opticks/analytic/sc.py:300} INFO - save_extras /tmp/blyth/opticks/tgltf/extras  : saved 248 
    /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf


     cat /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf | python -m json.tool




/tmp/blyth/opticks/tgltf/tgltf-gdml--.pretty.gltf the boundary spec are in nodes extras::

    3234     "nodes": [
    3235         {
    3236             "children": [
    3237                 1,
    3238                 3146
    3239             ],
    3240             "extras": {
    3241                 "boundary": "Vacuum///Rock"
    3242             },

    3243             "matrix": [
    3244                 -0.5431744456291199,
    ....
    3259                 1.0
    3260             ],
    3261             "mesh": 0,
    3262             "name": "ndIdx:  0,soIdx:  0,lvName:/dd/Geometry/Sites/lvNearSiteRock0xc030350"
    3263         },



Currently no surface spec::

    simon:opticksnpy blyth$ grep boundary /tmp/blyth/opticks/tgltf/tgltf-gdml--.pretty.gltf | sort | uniq
                    "boundary": "Acrylic///Air"
                    "boundary": "Acrylic///Aluminium"
                    "boundary": "Acrylic///GdDopedLS"
                    "boundary": "Acrylic///LiquidScintillator"
                    "boundary": "Acrylic///Nylon"
                    "boundary": "Acrylic///StainlessSteel"
                    "boundary": "Acrylic///Vacuum"
                    "boundary": "Air///Acrylic"
                    "boundary": "Air///Air"
                    "boundary": "Air///Aluminium"
                    "boundary": "Air///ESR"
                    "boundary": "Air///Iron"
                    "boundary": "Air///MixGas"
                    "boundary": "Air///PPE"
                    "boundary": "Air///StainlessSteel"
                    "boundary": "Aluminium///Co_60"
                    "boundary": "Aluminium///Foam"
                    "boundary": "Aluminium///Ge_68"
                    "boundary": "Bakelite///Air"
                    "boundary": "DeadWater///ADTableStainlessSteel"
                    "boundary": "DeadWater///Tyvek"
                    "boundary": "Foam///Bakelite"
                    "boundary": "IwsWater///ADTableStainlessSteel"
                    "boundary": "IwsWater///IwsWater"
                    "boundary": "IwsWater///PVC"
                    "boundary": "IwsWater///Pyrex"
                    "boundary": "IwsWater///StainlessSteel"
                    "boundary": "IwsWater///UnstStainlessSteel"
                    "boundary": "IwsWater///Water"
                    "boundary": "LiquidScintillator///Acrylic"
                    "boundary": "LiquidScintillator///GdDopedLS"
                    "boundary": "LiquidScintillator///Teflon"
                    "boundary": "MineralOil///Acrylic"


analytic/sc.py::

    034 class Nd(object):
     35     def __init__(self, ndIdx, soIdx, transform, boundary, name, depth, scene):
     36         """
     37         :param ndIdx: local within subtree nd index, used for child/parent Nd referencing
     38         :param soIdx: local within substree so index, used for referencing to distinct solids/meshes
     39         """
     40         self.ndIdx = ndIdx
     41         self.soIdx = soIdx
     42         self.transform = transform
     43         self.extras = dict(boundary=boundary)

    090 class Sc(object):
     91     def __init__(self, maxcsgheight=4):
    ...
    144     def add_node(self, lvIdx, lvName, soName, transform, boundary, depth):
    145 
    146         mesh = self.add_mesh(lvIdx, lvName, soName)
    147         soIdx = mesh.soIdx
    148 
    149         ndIdx = len(self.nodes)
    150         name = "ndIdx:%3d,soIdx:%3d,lvName:%s" % (ndIdx, soIdx, lvName)
    151 
    152         #log.info("add_node %s " % name)
    153         assert transform is not None
    154 
    155         nd = Nd(ndIdx, soIdx, transform, boundary, name, depth, self )
    156         nd.mesh = mesh
    ...
    166     def add_node_gdml(self, node, depth, debug=False):
    167 
    168         lvIdx = node.lv.idx
    169         lvName = node.lv.name
    170         soName = node.lv.solid.name
    171         transform = node.pv.transform
    172         boundary = node.boundary
    173         nodeIdx = node.index
    174 
    175         msg = "sc.py:add_node_gdml nodeIdx:%4d lvIdx:%2d soName:%30s lvName:%s " % (nodeIdx, lvIdx, soName, lvName )
    176         #print msg
    177 
    178         if debug:
    179             solidIdx = node.lv.solid.idx
    180             self.ulv.add(lvIdx)
    181             self.uso.add(solidIdx)
    182             assert len(self.ulv) == len(self.uso)
    183             sys.stderr.write(msg+"\n" + repr(transform)+"\n")
    184         pass
    185 
    186         nd = self.add_node( lvIdx, lvName, soName, transform, boundary, depth )


analytic/treebase.py::

    040 class Node(object):
    ...
    168     def _get_boundary(self):
    169         """
    170         ::
    171 
    172             In [23]: target.lv.material.shortname
    173             Out[23]: 'StainlessSteel'
    174 
    175             In [24]: target.parent.lv.material.shortname
    176             Out[24]: 'IwsWater'
    177 
    178 
    179         What about root volume
    180 
    181         * for actual root, the issue is mute as world boundary is not a real one
    182         * but for sub-roots maybe need use input, actually its OK as always parse 
    183           the entire GDML file
    184 
    185         """
    186         omat = 'Vacuum' if self.parent is None else self.parent.lv.material.shortname
    187         osur = ""
    188         isur = ""
    189         imat = self.lv.material.shortname
    190         return "/".join([omat,osur,isur,imat])
    191     boundary = property(_get_boundary)


* surf not imp



Contrast with G4DAE/Assimp route 
----------------------------------------

* hmm are going to need to use the G4DAE optical props anyhow... so 
  no point at moment to implement python parsing of G4DAE.  Actually 
  no point in long run of doing this either, the correct solution is 
  to add the missing info to the GDML. 

* need to find an appropriate point to ensure the GLTF and G4DAE trees
  are aligned, and then bring over the information missing ? 

  * ggeo/GScene is the likely location, its here that the G4DAE info is currently cleared 
  * perhaps having two GGeo instances (for the different routes) is the way to proceed ?
    (not so keen, seems too fundamental a change on first thought : but actually 
    when one is subbordinate it wouldnt be too disruptive)

  * hmm GScene has for the analytic route usurped a lot of what GGeo does for the triangulated

  * so the task is GGeo merging ...


* Hmm is bringing over even needed ... will need to merge GLTF 
  and G4DAE/GGeo info in the conversion to GPU geometry  



Analogous paths in the two routes
-------------------------------------

ggeo/GScene.cc::

    167 GSolid* GScene::createVolume(nd* n)
    168 {
    ...
    197 
    198     GSolid* solid = new GSolid(node_idx, gtransform, mesh, UINT_MAX, NULL );
    199 
    200     solid->setLevelTransform(ltransform);
    201 
    202     // see AssimpGGeo::convertStructureVisit
    203 
    204     solid->setSensor( NULL );
    205 
    206     solid->setCSGFlag( csg->getRootType() );
    207 
    208     solid->setCSGSkip( csg->isSkip() );
    209 
    210 
    211     // analytic spec currently missing surface info...
    212     // here need 
    213  
    214     unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing
    215 
    216     solid->setBoundary(boundary);     // unlike ctor these create arrays


assimprap/AssimGGeo.cc::

    0836 GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* /*parent*/)
     837 {
     ...
     912     GSolid* solid = new GSolid(nodeIndex, gtransform, mesh, UINT_MAX, NULL ); // sensor starts NULL
     913     solid->setLevelTransform(ltransform);
     914 
     915     const char* lv   = node->getName(0);
     916     const char* pv   = node->getName(1);
     917     const char* pv_p   = pnode->getName(1);
     918 
     919     gg->countMeshUsage(msi, nodeIndex, lv, pv);
     920 
     921     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
     922     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
     923     GSkinSurface*   sks = gg->findSkinSurface(lv);
     924 
    ....
     998     // boundary identification via 4-uint 
     999     unsigned int boundary = blib->addBoundary(
    1000                                                mt_p->getShortName(),
    1001                                                osurf ? osurf->getShortName() : NULL ,
    1002                                                isurf ? isurf->getShortName() : NULL ,
    1003                                                mt->getShortName()
    1004                                              );
    1005 
    1006     solid->setBoundary(boundary);
    1007     {
    1008        // sensor indices are set even for non sensitive volumes in PMT viscinity
    1009        // TODO: change that 
    1010        // this is a workaround that requires an associated sensitive surface
    1011        // in order for the index to be provided
    1012 
    1013         unsigned int surface = blib->getOuterSurface(boundary);
    1014         bool oss = slib->isSensorSurface(surface);
    1015         unsigned int ssi = oss ? NSensor::RefIndex(sensor) : 0 ;
    1016         solid->setSensorSurfaceIndex( ssi );
    1017     }

    0361 void AssimpGGeo::convertMaterials(const aiScene* scene, GGeo* gg, const char* query )
     362 {
     363     LOG(info)<<"AssimpGGeo::convertMaterials "
     364              << " query " << query
     365              << " mNumMaterials " << scene->mNumMaterials
     366              ;
     367 
     368     //GDomain<float>* standard_domain = gg->getBoundaryLib()->getStandardDomain(); 
     369     GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain();
     370 
     371 
     372     for(unsigned int i = 0; i < scene->mNumMaterials; i++)
     373     {
     374         unsigned int index = i ;  // hmm, make 1-based later 
     375 
     376         aiMaterial* mat = scene->mMaterials[i] ;
     377 
     378         aiString name_;
     379         mat->Get(AI_MATKEY_NAME, name_);
     380 
     381         const char* name = name_.C_Str();
     382 
     383         //if(strncmp(query, name, strlen(query))!=0) continue ;  
     384 
     385         LOG(debug) << "AssimpGGeo::convertMaterials " << i << " " << name ;
     386 
     387         const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
     388         const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );
     389 
     390         const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );
     391 
     392         const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
     393         const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
     394         const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
     395         const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
     396         const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );






