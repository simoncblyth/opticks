Opticks Analytic Scene Description
=====================================

ISSUE : Polygonization Repetition
----------------------------------

This is a symptom of not having an efficient scene representation yet.

::

    tboolean-0-polygonize


Currently NCSG::Polygonize blindly applies to every CSG node tree without:

* detection of same solid
* caching to avoid repeating work


gltf samples
------------

* https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/Lantern/glTF/Lantern.gltf


::

  "meshes": [
    {
      "primitives": [
        {
          "attributes": {
            "TEXCOORD_0": 0,
            "NORMAL": 1,
            "TANGENT": 2,
            "POSITION": 3
          },
          "indices": 4,
          "material": 0,
          "mode": 4
        }
      ],
      "name": "LanternPole_Body"
    },
    {
      "primitives": [
        {
          "attributes": {
            "TEXCOORD_0": 5,
            "NORMAL": 6,
            "TANGENT": 7,
            "POSITION": 8
          },
          "indices": 9,
          "material": 0,
          "mode": 4
        }
      ],
      "name": "LanternPole_Chain"
    },
    {
      "primitives": [
        {
          "attributes": {
            "TEXCOORD_0": 10,
            "NORMAL": 11,
            "TANGENT": 12,
            "POSITION": 13
          },
          "indices": 14,
          "material": 0,
          "mode": 4
        }
      ],
      "name": "LanternPole_Lantern"
    }
  ],



  "nodes": [
    {
      "children": [],
      "mesh": 0,
      "translation": [
        -3.82315421,
        13.01603,
        0.0
      ],
      "name": "LanternPole_Body"
    },
    {
      "children": [],
      "mesh": 1,
      "translation": [
        -9.582001,
        21.0378723,
        0.0
      ],
      "name": "LanternPole_Chain"
    },
    {
      "children": [],
      "mesh": 2,
      "translation": [
        -9.582007,
        18.0091515,
        0.0
      ],
      "name": "LanternPole_Lantern"
    },
    {
      "children": [
        0,
        1,
        2
      ],
      "scale": [
        0.06,
        0.06,
        0.06
      ],
      "translation": [
        0.237,
        -0.758,
        0.0
      ],
      "name": "Lantern"
    }
  ],
  "scene": 0,
  "scenes": [
    {
      "nodes": [
        3
      ]
    }
  ],





TODO : make input serialization smarter ? avoiding repetition 
---------------------------------------------------------------------

Input serialization growing from the test geometry route 
which has previously only handled small collections of volumes (eg 5 PMT solids).

Needs overhaul to handle

* material/surface/boundary assignment
* instances
* cache


Python CSG level analysis of the trees ?

* currently the solid for every node gets converted, `cn = solid.as_ncsg()`

  * instead use a higher level CSG node that can refer to a solid by index, 
    the solids living in a separate directory (like meshes in the old mesh-centric approach)

* howabout operating with separated lv too ?  

Ape the gdml structure in the NScene version::

    768     def init(self):
    769 
    770         self.materials = {}
    771         self.solids = {}
    772         self.volumes = {}
    773 
    774         for e in self.findall_("materials/material"):
    775             self.materials[e.name] = e
    776 
    777         for e in self.findall_("solids/*"):
    778             self.solids[e.name] = e
    779         pass
    780         for e in self.findall_("structure/*"):
    781             self.volumes[e.name] = e
    782         pass
    783         self.worldvol = self.elem.find("setup/world").attrib["ref"]
    784 



~/opticks/tests/tboolean_gdml.py::


     51     gdml = GDML.parse(gdmlpath)
     52     tree = Tree(gdml.world)
     53 
     54     subtree = tree.subtree(gsel, maxdepth=gmaxdepth, maxnode=gmaxnode, idx=gidx)
     55 
     56     log.info(" subtree %s nodes " % len(subtree) )
     57 
     58     cns = []
     59 
     60     for i, node in enumerate(subtree):
     61 
     62         solid = node.lv.solid
     63 
     64         if i % 100 == 0:log.info("[%2d] converting solid %r " % (i,solid.name))
     65 
     66         polyconfig = PolyConfig(node.lv.shortname)
     67 
     68         cn = solid.as_ncsg()
     69 
     70         has_name = cn.name is not None and len(cn.name) > 0
     71         assert has_name, "\n"+str(solid)
     72 
     73         if i > 0: # skip first node transform which is placement of targetNode within its parent 
     74             cn.transform = node.pv.transform
     75         pass
     76         cn.meta.update(polyconfig.meta )
     77         cn.meta.update(node.meta)
     78 
     79         cn.boundary = args.testobject
     80         cns.append(cn)
     81     pass
     ..
     84     container = CSG("box")
     85     container.boundary = args.container
     86     container.meta.update(PolyConfig("CONTAINER").meta)
     87 
     88     objs = []
     89     objs.append(container)
     90     objs.extend(cns)
     91 
     92     #for obj in objs: obj.dump()
     93 
     94     CSG.Serialize(objs, args.csgpath, outmeta=True )


::

    234     @classmethod
    235     def Serialize(cls, trees, base, outmeta=True):
    236         assert type(trees) is list
    237         assert type(base) is str and len(base) > 5, ("invalid base directory %s " % base)
    238         base = os.path.expandvars(base)
    239         log.info("CSG.Serialize : writing %d trees to directory %s " % (len(trees), base))
    240         if not os.path.exists(base):
    241             os.makedirs(base)
    242         pass
    243         for it, tree in enumerate(trees):
    244             treedir = cls.treedir(base,it)
    245             if not os.path.exists(treedir):
    246                 os.makedirs(treedir)
    247             pass
    248             tree.save(treedir)
    249         pass
    250         boundaries = map(lambda tree:tree.boundary, trees)
    251         cls.CheckNonBlank(boundaries)
    252         open(cls.txtpath(base),"w").write("\n".join(boundaries))
    253 
    254         if outmeta:
    255             meta = dict(mode="PyCsgInBox", name=os.path.basename(base), analytic=1, csgpath=base)
    256             meta_fmt_ = lambda meta:"_".join(["%s=%s" % kv for kv in meta.items()])
    257             print meta_fmt_(meta)  # communicates to tboolean--
    258         pass






::

    void test_Polygonize(const char* basedir, int verbosity, std::vector<NCSG*>& trees)
    {
        int rc0 = NCSG::Deserialize(basedir, trees, verbosity );  // revive CSG node tree for each solid
        assert(rc0 == 0 );

        int rc1 = NCSG::Polygonize(basedir, trees, verbosity );
        assert(rc1 == 0 );
    }







TODO: Converting GDML description into instance-ized CSG trees "OpticksSceneGraph"
-----------------------------------------------------------------------------------
   
CSG node trees are intended to describe individual "solids"
not entire scenes.  These need to be combines into
an OpticksSceneGraph format/serialization.

This is similar to the conversion of G4DAE/COLLADA trees 
into GPU geometries. But as starting from source GDML tree, 
can do a more complete job.

* use instancing for *all* solids (ie for all distinct shapes)
  minimizing the GPU memory requirements
  
  * ggeo analyses the G4DAE node tree to find
    repeated geometry ... this works but when have 
    direct access to the source GDML tree presumably 
    can do better by directly accessing all distinct shapes, 
    making CSG trees for each of them 

  * unsure how good GDML is at avoiding repetion, suspect 
    that some digesting will be needed 

  * polygonize the CSG trees into meshes, serialize and
    persist them together with the source CSG trees

    Currently with test geometry the meshes are not 
    persisted, just directly uploaded to GPU/OpenGL, but 
    when handling full geometries need to work with 
    a geocache serialization to avoid repeating work.

* construct scene graph structure (and serialization)
  aggregating references to the csg tree instances 
  together with their transforms

  * review OptiX geometry handling and OpenGL instancing, as currently 
    used to see how best to structure this to be 
    easily uploaded to GPU 


Whats needed in OpticksSceneGraph ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple buffer layout, for GPU consumption, be guided by customers:

* OptiX geometry instancing
* OpenGL geometry instancing

For each instance (perhaps uint4 buffer)

* unsigned index reference to CSG tree,  
* unsigned index reference to transform 
* identity code or reference to identity  

What to do different from current GGeo ?

* GGeo is mesh-centric, aim for instance-centric 
* design with simple serialization directory layout in mind 
* defer concatenation into big buffers as late as possible,
  retaining structure in directories for easy debug 


GDML->GGeo vs G4DAE->GGeo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So the process of converting GDML description, needs to 
follow a very similar course to the conversion of G4DAE 
COLLADA into a GPU description (GGeo and OGeo).

Do this inside GGeo ? Or another package ?

* initially start in GGeo and see how it goes
* recall GGeo was intended as a dumb substrate initially ...

The tasks are the same, so regard it as improving GGeo, 
not doing something new.


Validation
~~~~~~~~~~~

* implement in cfg4- OpticksSceneGraph -> G4 conversion, so 
  can compare two routes for geometry 

  * GDML -> G4 
  * GDML -> OpticksSceneGraph -> G4   


OpticksSceneGraph Technicalites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See scene- for examples of scene descriptions 

* use structure similar to gltf- 

* use python for parsing GDML rather than working in C++ with the G4 parse ? 
  Then can start from the (pmt-) dd.py detdesc/lxml parse 
  and bring it over to work with GDML 
    
* no reason why not to use python for input geometry conversion, 
  as in production this is only done once for each geometry 

  * can always migrate the python to C++ with some minimal XML parser external
    if it proves inconvenient to require python preprocessing 

Multi-level approach similar to NCSG chain, perhaps steered with 
an "NScene" ?  

* python prepares input serialization from the GDML, 
  finding all distinct shapes and writing CSG tree serializations
  for them,  
  (directory structure of .npy .json .txt)

* npy- embellishes the directory structure 
  eg using NPolygonization to write meshes into directory tree

* ggeo-  intermediate GPU geometry prep, however
  as have more control over NScene than with the COLLADA/Assimp/GGeo
  route expect will need less action at GGeo level  

* oglrap- to OpenGL

* ogeo-  to OptiX


Why not parse GDML with G4 and work with G4 in-memory tree ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* prefer to keep G4 dependency to a minimum as this yields more generally usable code
* promotes an independent approach 
* avoids having to work with G4 too much 




