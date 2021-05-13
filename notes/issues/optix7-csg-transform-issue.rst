optix7-csg-transform-issue
=============================


OptiX 5 on Darwin (and OptiX 6 on Linux) using Six.cc gives expected shapes with --one_gas_ias option::

   OGI=0 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=0 CAM=1 EYE=-1,0,0,1 TMIN=0.8 ./cxr.sh 
   OGI=0 CAM=0 EYE=-0.5,0,0,1 TMIN=0.4 ./cxr.sh 

   OGI=1 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=2 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=3 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=4 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=5 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=6 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=7 CAM=1 EYE=-2,0,0,1 ./cxr.sh 
   OGI=8 CAM=1 EYE=-2,0,0,1 EMM=t0 ./cxr.sh 
   OGI=9 CAM=1 EYE=-2,0,0,1  ./cxr.sh 


Because precisely the same CSGFoundry cache 
is being used for both the 6 and 7 renders the 
problem is isolated to how the IAS/GAS/SBT used by 7 
is interpreting that geometry. 
Most likely source of problems is transform referencing.

 



