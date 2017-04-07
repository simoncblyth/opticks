CSG Transform Support
=========================


CSG Tree Objective
----------------------

* recall the CSG trees are intended to be small per-solid trees
  corresponding to shape definitions (ie not definitions of full scene geometry)



Transforms with Ray-trace and SDF
------------------------------------

To translate or rotate a surface modeled as an CSG tree, 
apply the inverse transformation to the point for SDFs or the ray for 
raytracing before doing the SDF distance calc or ray tracing intersection
calc.


SDF
------

* Where to hold the transform in nnode trees and CSG trees ?

 * G4 allows the RHS of a boolean combination to be transformed using 
   a transform that lives with the combination



* use glm::mat4 ?


local/global transforms ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    111 double nunion::operator()(double px, double py, double pz)
    112 {

    ///    just transform px,py,pz here only  ?

    113     assert( left && right );
    114     double l = (*left)(px, py, pz) ;
    115     double r = (*right)(px, py, pz) ;
    116     return fmin(l, r);
    117 }


Perhaps can just locally apply the transform ? to the coordinates
passed down the tree ? Relying on subsequent transforms transforming 
again the transformed coordinates... this would be simplest.

The alternative would be to traverse up the tree thru parent 
links collecting and multiplying transforms and store that 
as a global transfrom within each node to apply to global coordinates.

Actually its not clear how to use global transforms as the evaluation is done
treewise ... with each node not knowing where it is in the tree ?

BUT: for internal nodes the coordinates are not actually used, they are 
just being passed down the tree until reach the leaves/primitives ... so this 
means can collect ancestor transforms into the primitives : this is 
what will need to do on GPU, so actually its better to take same approach on CPU 


* adopted globaltransform held in primitive, which is obtained at deserialization (in NCSG)
  from product of ancestor node transforms


Transforms GPU side 
--------------------

* does GPU need *tr* OR perhaps only *irit* will do, as primary action 
  is transforming impinging rays not directly geometry 


* optix Matrix4x4 uses row-major, Opticks standard follows OpenGL : column-major

::

    9.005 Are OpenGL matrices column-major or row-major?

    For programming purposes, OpenGL matrices are 16-value arrays with base vectors
    laid out contiguously in memory. The translation components occupy the 13th,
    14th, and 15th elements of the 16-element matrix, where indices are numbered
    from 1 to 16 as described in section 2.11.2 of the OpenGL 2.1 Specification.

    Column-major versus row-major is purely a notational convention. Note that
    post-multiplying with column-major matrices produces the same result as
    pre-multiplying with row-major matrices. The OpenGL Specification and the
    OpenGL Reference Manual both use column-major notation. You can use any
    notation, as long as it's clearly stated.


::

    /Developer/OptiX/include/optixu/optixu_matrix_namespace.h

    100   template <unsigned int M, unsigned int N>
    101   class Matrix
    102   {
    103   public:
    ...
    169   private:
    170       /** The data array is stored in row-major order */
    171       float m_data[M*N];
    172   };
    173 
       
    421   // Multiply matrix4x4 by float4
    422   OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const Matrix<4,4>& m, const float4& vec )
    423   {
    424     float4 temp;
    425     temp.x  = m[ 0] * vec.x +
    426               m[ 1] * vec.y +
    427               m[ 2] * vec.z +
    428               m[ 3] * vec.w;
    429     temp.y  = m[ 4] * vec.x +
    430               m[ 5] * vec.y +
    431               m[ 6] * vec.z +
    432               m[ 7] * vec.w;
    433     temp.z  = m[ 8] * vec.x +
    434               m[ 9] * vec.y +
    435               m[10] * vec.z +
    436               m[11] * vec.w;
    437     temp.w  = m[12] * vec.x +
    438               m[13] * vec.y +
    439               m[14] * vec.z +
    440               m[15] * vec.w;
    441 
    442     return temp;
    443   }


    709   typedef Matrix<2, 2> Matrix2x2;
    710   typedef Matrix<2, 3> Matrix2x3;
    711   typedef Matrix<2, 4> Matrix2x4;
    712   typedef Matrix<3, 2> Matrix3x2;
    713   typedef Matrix<3, 3> Matrix3x3;
    714   typedef Matrix<3, 4> Matrix3x4;
    715   typedef Matrix<4, 2> Matrix4x2;
    716   typedef Matrix<4, 3> Matrix4x3;
    717   typedef Matrix<4, 4> Matrix4x4;
    718 




Transforming BBox ?
---------------------

* http://dev.theomader.com/transform-bounding-boxes/
* http://www.cs.unc.edu/~zhangh/technotes/bbox.pdf

* https://www.geometrictools.com/Documentation/AABBForTransformedAABB.pdf
* https://github.com/erich666/GraphicsGems/blob/master/gems/TransBox.c
* http://www.akshayloke.com/2012/10/22/optimized-transformations-for-aabbs/



Models
-------

* input python model opticks.dev.csg.csg.CSG
* numpy array serialization
* NCSG created nnode model  


Where to hang the transform ?
--------------------------------

parent.rtransform OR node.transform ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* transform reference on CSG operation node is advantageous, as no space pressure there

  * actually above "advantage" is conflating the serialization with the in memory nnode model, 
    the in nnode model does not have any space issues, and it does not need to 
    precisely follow what the serialization does

* so can define and serialize using rtransform and then deserialize onto transforms 
  directly on nodes as that is easier in usage 

* not so clear that node.transform is easier in usage... as 
  would mean that every primitive needs to implement coordinate transformations 
  handling as opposed to just the 3 CSG operation nodes



