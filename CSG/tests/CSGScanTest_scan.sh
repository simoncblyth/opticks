#!/bin/bash 
usage(){ cat << EOU
CSGScanTest_scan.sh
====================

~/o/CSG/tests/CSGScanTest_scan.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))


NAMES=$(sed 's/#.*//' << EON
JustOrb
BoxedSphere
ZSphere
Cone
Hyperboloid
Box3
#Plane
#Slab
Cylinder
OldCylinder
Disc
ConvexPolyhedronCube
ConvexPolyhedronTetrahedron
Ellipsoid
UnionBoxSphere
UnionListBoxSphere
UnionLLBoxSphere
IntersectionBoxSphere
#OverlapBoxSphere
#OverlapThreeSphere
ContiguousThreeSphere
DiscontiguousThreeSphere
DiscontiguousTwoSphere
#ContiguousBoxSphere
DiscontiguousBoxSphere
DifferenceBoxSphere
ListTwoBoxTwoSphere
RotatedCylinder
DifferenceCylinder
InfCylinder
InfPhiCut
InfThetaCut
InfThetaCutL
BoxSubSubCylinder
EON
)


#for name in $NAMES ; do printf "%s\n" $name ; done

for name in $NAMES  
do
     printf "%s\n" $name 

     #GEOM=$name ./CSGScanTest.sh info
     GEOM=$name ./CSGScanTest.sh run
 
done


