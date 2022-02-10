GeneralSphereDEV
===================



Curious no isect from genstep IXYZ (0,0,0)
---------------------------------------------

::


    In [15]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == 0) ]                                                                                
    Out[15]: array([], shape=(0, 4), dtype=int8)

    In [16]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == 1) ]                                                                                
    Out[16]: 
    array([[ 0,  0,  1,  3],
           [ 0,  0,  1,  5],
           [ 0,  0,  1, 11],
           [ 0,  0,  1, 14],
           [ 0,  0,  1, 20],
           [ 0,  0,  1, 33],
           [ 0,  0,  1, 43],
           [ 0,  0,  1, 52],
           [ 0,  0,  1, 53],
           [ 0,  0,  1, 54],
           [ 0,  0,  1, 57],
           [ 0,  0,  1, 87]], dtype=int8)

    In [17]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == -1) ]                                                                               
    Out[17]: 
    array([[ 0,  0, -1,  4],
           [ 0,  0, -1, 16],
           [ 0,  0, -1, 32],
           [ 0,  0, -1, 46],
           [ 0,  0, -1, 54],
           [ 0,  0, -1, 56],
           [ 0,  0, -1, 57],
           [ 0,  0, -1, 77],
           [ 0,  0, -1, 86],
           [ 0,  0, -1, 99]], dtype=int8)


Using regular bicycle spoke photon directions with negative PHO 
----------------------------------------------------------------------


Unexpected miss in outer direction around XY plane::

     IXYZ=-5,0,0 PHO=-100 ./csg_geochain.sh 

Seems like photons with angle in the thetacut range failing to land. 

* need some special handling of unbounded 


CSG/csg_intersect_leaf.h::

    1485     if(complement)  // flip normal, even for miss need to signal the complement with a -0.f  
    1486     {
    1487         isect.x = -isect.x ;
    1488         isect.y = -isect.y ;
    1489         isect.z = -isect.z ;
    1490     }
    1491     /*
    1492 
    1493     // unbounded leaf cutters MISS need some special casing 
    1494     // unbounded MISS needs to be converted into EXIT but only in some directions 
    1495 
    1496     else if(valid_isect == false && typecode == CSG_THETACUT)
    1497     {
    1498         isect.x = -isect.x ;
    1499     }
    1500     */
    1501     
    1502     return valid_isect ;
    1503 }


The above is too indescriminate, need to do it only when the rays are headed for the 
otherside at infinity. 

TODO: try not constructing by intersecting but instead by intersecting with the complemented other side 




Using 2D(embedded in 3D) cross products : can determine if ray direction is between cone directions
---------------------------------------------------------------------------------------------------------



::

     IXYZ=9,0,9 ./csg_geochain.sh ana        # expected
     IXYZ=10,0,10 ./csg_geochain.sh ana      # no isect : duh : because there are no gensteps at that grid position



Debug blank cxr_geochain.sh render with thetacut : NOW FIXED 
----------------------------------------------------------------

Modify to a full sphere to orientate and prepare debug. 
Set EYE inside the sphere so every pixel should intersect::

    EYE=-0.25,0,0 ./cxr_geochain.sh 

Get logging::

    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3494 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3622 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3749 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2852 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2981 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3109 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3238 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2333 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2463 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2593 


Now back to thetacut sphere and recreate the CSG geom::

   x4 ; vi GeneralSphereDEV.sh 
   gc ; ./run.sh 

Huh, working already. Must have been just that the was not seeing the new headers::

    EYE=-1,-1,1 TMIN=0.1 ./cxr_geochain.sh 



Onwards to phicut 
--------------------

Pacman, but failing to intersect with half of phi:: 

    IXYZ=-3,3,0 ./csg_geochain.sh 

