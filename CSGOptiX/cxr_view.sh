#!/bin/bash -l 

usage(){ cat << EOU

    EYE=0,-0.5,0.75,1 TMIN=0.5 MOI=Hama:0:1000 ./cxr_view.sh 
    
    MOI=Hama:0:1000 ./cxr_view.sh 

    MOI=NNVT:0:1000 ./cxr_view.sh 

    MOI=NNVT:0:1000 EYE=-10,-10,-10,1 ./cxr_view.sh 

    
    MOI=NNVT:0:1000 EYE=0,2,-4 ./cxr_view.sh 





   MOI=sWorld EYE=0,0.6,0.4 TMIN=0.4 ./cxr_view.sh


Nice views::

    MOI=NNVT:0:1000 EYE=0,1,-2,1 ./cxr_view.sh 

    MOI=NNVT:0:1000 EYE=0,2,-4,1 ./cxr_view.sh 

    MOI=sWaterTube EYE=0,1,-0.5 LOOK=0,0,-0.5 ./cxr_view.sh 

    MOI=sWaterTube EYE=0,1,-0.5 LOOK=0,0,-0.5 TMIN=1 ./cxr_view.sh 



    MOI=sWaterTube EYE=0,1,-1,1 LOOK=0,0,-1 ./cxr_view.sh 


RTP tangential::

   MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   GDB=gdb MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   EYE=-1,0,0 MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   EYE=0,0,1 UP=0,-1,0  MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   EYE=0,0,2 UP=0,-1,0 TMIN=0.1  MOI=solidXJfixture:10:-3 ./cxr_view.sh 


   Radial outwards as UP is quite natural 

   EYE=0,1,1 UP=1,0,0 TMIN=0.1  MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   EYE=1,1,1 UP=1,0,0 TMIN=0.1  MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   EYE=2,1,1 UP=1,0,0 TMIN=0.1 CAM=1  MOI=solidXJfixture:10:-3 ./cxr_view.sh 

   EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:10:-3 ./cxr_view.sh 


   EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:55:-3 ./cxr_view.sh 


Mid chimney fixture::


              R
              |
              +-- P
             /
            T
           

   EYE=0,-1,0 UP=1,0,0 TMIN=0.1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 

   EYE=0,-4,0 UP=1,0,0 TMIN=0.1 MOI=solidXJfixture:0:-3 ./cxr_view.sh  


   EYE=0,-4,0 UP=1,0,0 TMIN=0.1 MOI=solidXJfixture:0:-3 ./cxr_view.sh  




   EYE=4,-2,-2 UP=1,0,0 TMIN=0.1 MOI=solidXJfixture:2:-3 ./cxr_view.sh  
   EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 MOI=solidXJfixture:2:-3 ./cxr_view.sh  

   EYE=16,-8,-8 UP=0,-1,0 TMIN=0.1 MOI=solidXJfixture:2:-3 ./cxr_view.sh  
        sticks and hatboxes : whacky angle   



   EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:41:-3 ./cxr_view.sh




   EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh
        this reproduces the view from the image grid of 64

   EYE=8,-4,-4 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh
        back up, see the chimney cylinder with lots of coincidence speckle
        but looses sight of the fixture

   EYE=0,-4,0 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        try tangential view, japanese temple with speckle behind

   EYE=0,-4,0 UP=1,0,0 TMIN=4 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        tangential view, upping TMIN 

   EYE=0,-8,0 UP=1,0,0 TMIN=8 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        tangential view, upping TMIN and backing away, see confusing close view of sticks to right   

   EYE=0,-16,0 UP=1,0,0 TMIN=16 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        tangential view, upping TMIN and backing away, makes more sense now than can see multiple
        sticks, one with its cover cut away. Also can now see the curve of the sphere. 

   EYE=0,-32,0 UP=1,0,0 TMIN=32 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        now can see the hatboxes, an interesting view showing context of the fixture, 
        but the parallel projection makes it kinda wierd but good at understandable 
        cutting of geometry.
        Also the bottom half of the frame is just a single block of color
        TODO: make some more like this from a bit higher up in R

   EYE=16,-32,0 LOOK=16,0,0 UP=1,0,0 TMIN=32 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Look at a point 16 extents above the fixture

   EYE=32,-32,0 LOOK=32,0,0 UP=1,0,0 TMIN=32 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Look at a point 32 extents above the fixture

   EYE=32,-48,0 LOOK=32,0,0 UP=1,0,0 TMIN=48 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Look at a point 32 extents above the fixture

   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh sp | grep 32,-48 
   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh cp | grep 32,-48  | sh 
   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh s5 | grep 32,-48 

   EYE=32,-48,0 LOOK=32,0,0 UP=1,0,0 TMIN=48 CAM=1 ZOOM=0.25 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Try zoom out  
 
   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh cp | grep zoom_0.25 | sh 


   EYE=0,-32,0 UP=1,0,0 TMIN=32 CAM=0 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Try with perspective cam, its easier to understand 

   EYE=8,-32,0 UP=1,0,0 TMIN=32 CAM=0 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Try moving up a bit in R, good view

   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh sp | grep 8,-32 
   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh cp | grep 8,-32 
   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh cp | grep 8,-32 | sh
   GEOM=solidXJfixture:0:-3 ./cxr_pub.sh s5 | grep 8,-32

      publish into s5 

   EYE=4,-8,0 UP=1,0,0 TMIN=8 CAM=0 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
       Try a closer followup 



   EYE=8,-16,0 UP=1,0,0 TMIN=16 CAM=0 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        Try moving in, ok not a good at showing context as previous 




   EYE=-4,-4,0 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        sink down in R to look up at fixture, see it with swash of chimney cyl 

   EYE=-4,0,0 UP=0,1,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        try directly up view : see it but no chimney edge

   EYE=-10,0,0 UP=0,1,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        backing away can see chimney ring of very slightly different shade

   EYE=-10,0,4 UP=0,1,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:0:-3 ./cxr_view.sh 
        offset a bit, get the swash again


   EYE=2,-1,-1 UP=1,0,0 TMIN=0.1 CAM=1 MOI=solidXJfixture:2:-3 ./cxr_view.sh
        this reproduces the view from the image grid of 64

   EYE=4,-2,-2 UP=1,0,0 TMIN=0.0 CAM=1 MOI=solidXJfixture:2:-3 ./cxr_view.sh
        backing up shows that its just a funny angle on cut stick base

   EYE=8,-4,-4 UP=1,0,0 TMIN=0.0 CAM=1 MOI=solidXJfixture:2:-3 ./cxr_view.sh
        backing up more makes it plain, the cause of the curious shape 
        is the tmin cutting off the top of sticks



EOU
}


#moi=sStrut      # what to look at 
moi=sWaterTube   # should be same as lLowerChimney_phys
emm=t0      # "t0" : tilde zero meaning all       "t0," : exclude bit 0 global,  "t8," exclude mm 8 
zoom=1
eye=-1,-1,-1,1
tmin=0.4
cam=0
quality=90

export MOI=${MOI:-$moi}
export EMM=${EMM:-$emm}
export ZOOM=${ZOOM:-$zoom}
export EYE=${EYE:-$eye}
export TMIN=${TMIN:-$tmin} 
export CAM=${CAM:-$cam} 
export QUALITY=${QUALITY:-$quality} 

nameprefix=cxr_view_${sla}_

if [ -n "$EYE" ]; then 
   nameprefix=${nameprefix}_eye_${EYE}_
fi 
if [ -n "$LOOK" ]; then 
   nameprefix=${nameprefix}_look_${LOOK}_
fi 
if [ -n "$ZOOM" ]; then 
   nameprefix=${nameprefix}_zoom_${ZOOM}_
fi 
if [ -n "$TMIN" ]; then 
   nameprefix=${nameprefix}_tmin_${TMIN}_
fi 


export NAMEPREFIX=$nameprefix               # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export OPTICKS_RELDIR=cam_${CAM}_${EMM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)

export TOPLINE="./cxr_view.sh $MOI      # EYE $EYE LOOK $LOOK UP $UP      EMM $EMM  $stamp  $version " 

source ./cxr.sh     

exit 0

