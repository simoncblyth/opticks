ocsgbsp-src(){      echo externals/ocsgbsp.bash ; }
ocsgbsp-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ocsgbsp-src)} ; }
ocsgbsp-vi(){       vi $(ocsgbsp-source) ; }
ocsgbsp-env(){      olocal- ; opticks- ; }
ocsgbsp-usage(){ cat << EOU

CSG BSP
=========

* https://en.wikipedia.org/wiki/Binary_space_partitioning

Originally discounted CSG BSP due to prejudice against
it for producing horrible triangles... 
but hybrid implicit/parametric is running into sticky subdiv territory. 
So perhaps could combine CSG BSP with a remeshing step afterwards
to clean up the tris.  

* https://github.com/simoncblyth/csgjs-cpp
* https://github.com/dabroz/csgjs-cpp
* https://github.com/evanw/csg.js/


Code Review
-------------

Public API
~~~~~~~~~~~~

::

     22 struct csgjs_vector
     23 {
     24     float x, y, z;
     25 
     26     csgjs_vector() : x(0.0f), y(0.0f), z(0.0f) {}
     27     explicit csgjs_vector(float x, float y, float z) : x(x), y(y), z(z) {}
     28 };
     29 
     30 struct csgjs_vertex
     31 {
     32     csgjs_vector pos;
     33     csgjs_vector normal;
     34     csgjs_vector uv;
     35 };
     36 
     37 struct csgjs_model
     38 {
     39     std::vector<csgjs_vertex> vertices;
     40     std::vector<int> indices;
     41 };
     42 
     43 // public interface - not super efficient, if you use multiple CSG operations you should
     44 // use BSP trees and convert them into model only once. Another optimization trick is
     45 // replacing csgjs_model with your own class.
     46 
     47 csgjs_model csgjs_union(const csgjs_model & a, const csgjs_model & b);
     48 csgjs_model csgjs_intersection(const csgjs_model & a, const csgjs_model & b);
     49 csgjs_model csgjs_difference(const csgjs_model & a, const csgjs_model & b);
     50 


Private
~~~~~~~~~~~

::

     63 // Represents a plane in 3D space.
     64 struct csgjs_plane
     65 {
     66     csgjs_vector normal;
     67     float w;
     68 
     69     csgjs_plane();
     70     csgjs_plane(const csgjs_vector & a, const csgjs_vector & b, const csgjs_vector & c);
     71     bool ok() const;
     72     void flip();
     73     void splitPolygon(const csgjs_polygon & polygon, std::vector<csgjs_polygon> & coplanarFront, std::vector<csgjs_polygon> & coplanarBack, std::vector<csgjs_polygon> & front, std::vector<csgjs_    polygon> & back) const;
     74 };
     75 
     76 // Represents a convex polygon. The vertices used to initialize a polygon must
     77 // be coplanar and form a convex loop. They do not have to be `CSG.Vertex`
     78 // instances but they must behave similarly (duck typing can be used for
     79 // customization).
     80 // 
     81 // Each convex polygon has a `shared` property, which is shared between all
     82 // polygons that are clones of each other or were split from the same polygon.
     83 // This can be used to define per-polygon properties (such as surface color).
     84 struct csgjs_polygon
     85 {
     86     std::vector<csgjs_vertex> vertices;
     87     csgjs_plane plane;
     88     void flip();
     89 
     90     csgjs_polygon();
     91     csgjs_polygon(const std::vector<csgjs_vertex> & list);
     92 };


    094 // Holds a node in a BSP tree. A BSP tree is built from a collection of polygons
     95 // by picking a polygon to split along. That polygon (and all other coplanar
     96 // polygons) are added directly to that node and the other polygons are added to
     97 // the front and/or back subtrees. This is not a leafy BSP tree since there is
     98 // no distinction between internal and leaf nodes.
     99 struct csgjs_csgnode
    100 {
    101     std::vector<csgjs_polygon> polygons;
    102     csgjs_csgnode * front;
    103     csgjs_csgnode * back;
    104     csgjs_plane plane;
    105 
    106     csgjs_csgnode();
    107     csgjs_csgnode(const std::vector<csgjs_polygon> & list);
    ///   ^^^^^^^ entry point ^^^^^^^^^^^^^    

    108     ~csgjs_csgnode();
    109 
    110     csgjs_csgnode * clone() const;
    111     void clipTo(const csgjs_csgnode * other);
    112     void invert();
    113     void build(const std::vector<csgjs_polygon> & polygon);
    114     std::vector<csgjs_polygon> clipPolygons(const std::vector<csgjs_polygon> & list) const;
    115     std::vector<csgjs_polygon> allPolygons() const;
    116 };


::

    531 inline static std::vector<csgjs_polygon> csgjs_modelToPolygons(const csgjs_model & model)
    532 {
    533     std::vector<csgjs_polygon> list;
    534     for (size_t i = 0; i < model.indices.size(); i+= 3)
    535     {
    536         std::vector<csgjs_vertex> triangle;
    537         for (int j = 0; j < 3; j++)
    538         {
    539             csgjs_vertex v = model.vertices[model.indices[i + j]];
    540             triangle.push_back(v);
    541         }
    542         list.push_back(csgjs_polygon(triangle));
    543     }
    544     return list;
    545 }


    566 inline static csgjs_model csgjs_operation(const csgjs_model & a, const csgjs_model & b, csg_function fun)
    567 {
    568     csgjs_csgnode * A = new csgjs_csgnode(csgjs_modelToPolygons(a));
    569     csgjs_csgnode * B = new csgjs_csgnode(csgjs_modelToPolygons(b));
    570     csgjs_csgnode * AB = fun(A, B);
    571     std::vector<csgjs_polygon> polygons = AB->allPolygons();
    572     delete A; A = 0;
    573     delete B; B = 0;
    574     delete AB; AB = 0;
    575     return csgjs_modelFromPolygons(polygons);
    576 }
    577 
    578 csgjs_model csgjs_union(const csgjs_model & a, const csgjs_model & b)
    579 {
    580     return csgjs_operation(a, b, csg_union);
    581 }

EOU
}

#ocsgbsp-edit(){ vi $(opticks-home)/cmake/Modules/FindCSGBSP.cmake ; }

ocsgbsp-url(){  echo https://github.com/simoncblyth/csgjs-cpp ; }
ocsgbsp-dir(){  echo $(opticks-prefix)/externals/csgbsp/csgjs-cpp ; }
ocsgbsp-bdir(){ echo $(opticks-prefix)/externals/csgbsp/csgjs-cpp.build ; }

ocsgbsp-cd(){  cd $(ocsgbsp-dir); }
ocsgbsp-bcd(){ cd $(ocsgbsp-bdir) ; }

ocsgbsp-get(){
   local iwd=$PWD
   local dir=$(dirname $(ocsgbsp-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d csgjs-cpp ] && git clone $(ocsgbsp-url)
   cd $iwd
}

ocsgbsp-cmake()
{
    local iwd=$PWD
    local bdir=$(ocsgbsp-bdir)
    local sdir=$(ocsgbsp-dir)

    mkdir -p $bdir
    rm -f "$bdir/CMakeCache.txt"

    ocsgbsp-bcd   
    opticks-

    cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       $* \
       $sdir

    cd $iwd
}

ocsgbsp-make()
{
    local iwd=$PWD
    ocsgbsp-bcd
    cmake --build . --config Debug --target ${1:-install}
    cd $iwd
}

ocsgbsp--()
{
   ocsgbsp-get
   ocsgbsp-cmake
   ocsgbsp-make all

   #if [ "$(uname)" == "Darwin" ]; then
   #    echo sleeping for 2s : see and env/tools/cmak.bash and https://gitlab.kitware.com/cmake/cmake/issues/16155
   #    sleep 2   
   #fi  
   
   ocsgbsp-make install
}

