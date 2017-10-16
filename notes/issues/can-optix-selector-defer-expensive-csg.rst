Can OptiX Selector Defer Expensive CSG ?
===========================================


Overview
---------

Ray tracing geometries composed of millions of triangles, or 
manually partitioned analytic geometries is substantially faster than
ray tracing CSG trees.

Perhaps CSG tree ray tracing can be deferred until actually needed by 
replacing the complicated CSG trees for PMT instances with just their
outer volumes for rays originating outside the bbox ?
 

OptiX Geometry Selector
-------------------------

A selector is similar to a group in that it is a collection of higher level
graph nodes. The number of nodes in the collection is set by
rtSelectorSetChildCount, and the individual children are assigned with
rtSelectorSetChild. Valid child types are rtGroup, rtGeometryGroup,
rtTransform, and rtSelector.

The main difference between selectors and groups is that selectors do not have
an acceleration structure associated with them. Instead, a visit program is
specified with rtSelectorSetVisitProgram. This program is executed every time a
ray encounters the selector node during graph traversal. The program specifies
which children the ray should continue traversal through by calling
rtIntersectChild.

A typical use case for a selector is dynamic (i.e. per-ray) level of detail: an
object in the scene may be represented by a number of geometry nodes, each
containing a different level of detail version of the object. The geometry
groups containing these different representations can be assigned as children
of a selector. The visit program can select which child to intersect using any
criterion (e.g. based on the footprint or length of the current ray), and
ignore the others.

As for groups and other graph nodes, child nodes of a selector can be shared
with other graph nodes to allow flexible instancing.


optixSelector sample
----------------------

Setup a pair of colocated sphere geometry groups with different radii, tie them together 
into a *Selector* with child count 2.

::

    226 void create_geometry( RTcontext context, RTmaterial material[] )
    227 {
    228     /* Setup two geometry groups */
    229 
    230     // Geometry nodes (two spheres at same position, but with different radii)
    231     RTgeometry geometry[2];
    232 
    233     geometry[0] = makeGeometry(context, 1);
    234     makeGeometryPrograms(context, geometry[0], "sphere.cu", "intersect", "bounds");
    235     makeGeometryVariable4f(context, geometry[0], "sphere", 0.0f, 0.0f, 0.0f, 0.5f);
    236 
    237     geometry[1] = makeGeometry(context, 1);
    238     makeGeometryPrograms(context, geometry[1], "sphere.cu", "intersect", "bounds");
    239     makeGeometryVariable4f(context, geometry[1], "sphere", 0.0f, 0.0f, 0.0f, 1.0f);
    240 
    241     // Geometry instance nodes
    242     RTgeometryinstance instance[2];
    243     instance[0] = makeGeometryInstance( context, geometry[0], material[0] );
    244     instance[1] = makeGeometryInstance( context, geometry[1], material[1] );
    245 
    246     // Accelerations nodes
    247     RTacceleration acceleration[2];
    248     acceleration[0] = makeAcceleration( context, "NoAccel" );
    249     acceleration[1] = makeAcceleration( context, "NoAccel" );
    250 
    251     // Geometry group nodes
    252     RTgeometrygroup group[2];
    253     group[0] = makeGeometryGroup( context, instance[0], acceleration[0] );
    254     group[1] = makeGeometryGroup( context, instance[1], acceleration[1] );
    255 
    256     /* Setup selector as top objects */
    257 
    258     // Init selector node
    259     RTselector selector;
    260     RTprogram  stor_visit_program;
    261     RT_CHECK_ERROR( rtSelectorCreate(context,&selector) );
    262     RT_CHECK_ERROR( rtProgramCreateFromPTXFile(context,ptxpath("selector_example.cu").c_str(),"visit",&stor_visit_program) );
    263     RT_CHECK_ERROR( rtSelectorSetVisitProgram(selector,stor_visit_program) );
    264     RT_CHECK_ERROR( rtSelectorSetChildCount(selector,2) );
    265     RT_CHECK_ERROR( rtSelectorSetChild(selector, 0, group[0]) );
    266     RT_CHECK_ERROR( rtSelectorSetChild(selector, 1, group[1]) );
    267 
    268     // Attach selector to context as top object
    269     RTvariable var_group;
    270     RT_CHECK_ERROR( rtContextDeclareVariable(context,"top_object",&var_group) );
    271     RT_CHECK_ERROR( rtVariableSetObject(var_group, selector) );
    272 }





Pick which geometry group to show based on ray direction::

     29 #include <optix.h>
     30 #include <optixu/optixu_math_namespace.h>
     31 
     32 using namespace optix;
     33 
     34 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     35 
     36 RT_PROGRAM void visit()
     37 {
     38   unsigned int index = (unsigned int)( ray.direction.y < 0.0f );
     39   rtIntersectChild( index );
     40 }



Hmm not clear how to structure 
----------------------------------


Looks like will need a separate selector for every instance... for instance identity::

      Selector
          GeometryGroup 
               GeometryInstance(Geometry,Material)
               Acceleration
          GeometryGroup 
               GeometryInstance(Geometry,Material)
               Acceleration





