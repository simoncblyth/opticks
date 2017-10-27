csg_bounds_gpu_bbox_not_csg_aware_causing_overlarge
=====================================================

ISSUE : GPU side bbox not CSG difference/intersection aware 
---------------------------------------------------------------

* getting overlarge union bbox with difference tree (presumably same problem with intersect)
* CPU side bounds have some rudimentary CSG awareness (done in nnode)
  makeing GPU bounds are greater than the CPU ones ...


* **NB: THIS IS NOT YET FIXED : NEED TO SET BBOX ON ROOT NODE**


Approach
------------

* port the CPU side tree bounds calc GPU side 

  * this is a non-starter, the algo is far too complicated (see below nnode::bbox)
  * also the algo is an approximation that perhaps can
    be replaced in future with a better one (eg parametric point cloud idea)

* copy from CPU ?

  * YEP : no choice
  * does the root node (which is either a single primitive or a CSG operation node) 
    have spare slots in quads 2,3 ? 



DONE
-----

* review the nnode CSG bbox calc


TODO
-----

* investigate conseqences of container impingement
* overlarge bbox reported to OptiX assumed to cause slower accel traversal
* vague recollection of overlarge bbox with torus, find notes on that


Observations
----------------

For tboolean-cyd the overlarge GPU bounds 
cause no problems it seems, presumably because 
no impingment with container.


tboolean-cyd example
-------------------------

::

    tboolean-;tboolean-cyd
    ...
    // intersect_analytic.cu:bounds buffer sizes pts:   4 pln:   0 trs:   6 
    // csg_bounds_prim CSG_FLAGNODETREE  primIdx   0 partOffset   0  numParts   3 -> height  1 -> numNodes  3  tranBuffer_size   6 
    // csg_bounds_prim CSG_FLAGNODETREE  primIdx   1 partOffset   3  numParts   1 -> height  0 -> numNodes  1  tranBuffer_size   6 
    // csg_bounds_cylinder center   0.000   0.000 (  0.000 =0)  radius 200.000 z1 -100.000 z2 100.000 
    // csg_intersect_primitive.h:csg_bounds_sphere  tbb.min (  -100.0000  -100.0000     0.0000 )  tbb.max (   100.0000   100.0000   200.0000 ) 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 0  bnd0:123 typ0:  3  min  -200.0000  -200.0000  -100.0000 max   200.0000   200.0000   200.0000 
    // intersect_analytic.cu:bounds.NODETREE primIdx: 1  bnd0:124 typ0:  6  min -1000.0000 -1000.0000 -1000.0000 max  1000.0000  1000.0000  1000.0000 

::

    tboolean-cyd(){ TESTCONFIG=$($FUNCNAME-) TORCHCONFIG=$($FUNCNAME-torch-) tboolean-- $* ; }
    tboolean-cyd-(){  $FUNCNAME- | python $* ; }  
    tboolean-cyd--(){ cat << EOP 
    import numpy as np
    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="$TMP/$FUNCNAME")

    CSG.boundary = args.testobject
    CSG.kwa = dict(verbosity="1", poly="IM", resolution="4" )

    container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution="4", verbosity="0" )

    ra = 200 
    z1 = -100
    z2 = 100

    a = CSG("cylinder", param=[0,0,0,ra], param1=[z1,z2,0,0] )
    b = CSG("sphere", param=[0,0,z2,ra/2]  )

    obj = a - b 

    CSG.Serialize([container, obj], args.csgpath )

    """  

    Expected bbox  (-200,-200,-100)  (200,200,100)

                    

                     _
                   .   .    
                 .       .
             +---.---+---.---+ (200,100) 
             |   .       .   |
             |     .   .     |   
         ----|-------^-------|---- X
             |               |   
             |               |   
             +---------------+

    """

    EOP
    }








nnode::bbox
--------------

::

     576 nbbox nnode::bbox() const
     577 {
     578     /*
     579     The gtransforms are applied at the leaves, ie the bbox returned
     580     from primitives already uses the full heirarchy of transforms 
     581     collected from the tree by *update_gtransforms()*.  
     582 
     583     Due to this it would be incorrect to apply gtransforms 
     584     of composite nodes to their bbox as those gtransforms 
     585     together with those of their progeny have already been 
     586     applied down at the leaves.
     587 
     588     Indeed without subnode bbox being in the same CSG tree top frame
     589     it would not be possible to combine them.
     590     */
     591 
     592     if(verbosity > 0)
     593     LOG(info) << "nnode::bbox " << desc() ;
     594 
     595     nbbox bb = make_bbox() ;
     596 
     597     if(is_primitive())
     598     {
     599         get_primitive_bbox(bb);
     600     }
     601     else
     602     {
     603         get_composite_bbox(bb);
     604     }
     605     return bb ;
     606 }

::

     466 void nnode::get_composite_bbox( nbbox& bb ) const
     467 {
     468     assert( left && right );
     469 
     470     bool l_unbound = left->is_unbounded();
     471     bool r_unbound = right->is_unbounded();
     472 
     473     bool lr_unbound = l_unbound && r_unbound ;
     474     if(lr_unbound)
     475     {
     476         LOG(warning) << "nnode::get_composite_bbox lr_unbound leave bb as is " ;
     477         return ;
     478     }
     479     //assert( !lr_unbound  && " combination of two unbounded prmitives is not allowed " );
     480 
     481 
     482     nbbox l_bb = left->bbox();
     483     nbbox r_bb = right->bbox();
     484 
     485 
     486     if( left->is_unbounded() )
     487     {
     488         assert(l_bb.is_empty());
     489         bb = r_bb ;
     490     }
     491     else if( right->is_unbounded() )
     492     {
     493         assert(r_bb.is_empty());
     494         bb = l_bb ;
     495     }
     496     else
     497     {
     498         if(left->is_primitive()) left->check_primitive_bb(l_bb);
     499         if(right->is_primitive()) right->check_primitive_bb(r_bb);
     500 
     501         nbbox::CombineCSG(bb, l_bb, r_bb, type, verbosity  );
     502     }
     503 
     504     if(verbosity > 0)
     505     std::cout << "nnode::composite_bbox "
     506               << " left " << left->desc()
     507               << " right " << right->desc()
     508               << " bb " << bb.desc()
     509               << std::endl
     510               ;
     511 
     512 } 


::

    288 void nbbox::CombineCSG(nbbox& comb, const nbbox& a, const nbbox& b, OpticksCSG_t op, int verbosity )
    289 {
    290 /*
    291 
    292 Obtaining the BBOX of a CSG tree is non-trivial
    293 ===================================================
    294 
    295 Alternative Approach
    296 ----------------------
    297 
    298 * perhaps these complications can be avoiding by forming a bbox
    299   from the composite parametric points (ie look at all parametric 
    300   points of all primitives transformed into CSG tree root frame and 
    301   make a selection based on their composite SDF values... points
    302   within epsilon of zero are regarded as being on the composite 
    303   surface). 
    304 
    305   As the parametric points should start exactly at SDF zero 
    306   for the primitives, and they are transformed only rather locally 
    307   I expect that a very tight epsilon 1e-5 should be appropriate.
    308 
    309 
    310 Analytic BB(CSG) approach
    311 ---------------------------
    312 
    313 * see csgbbox- for searches for papers to help with an algebra of CSG bbox 
    314   and a look at how OpenSCAD and povray handle this  
    315 
    316 * best paper found on this by far is summarised below
    317 
    318 
    319 Computing CSG tree boundaries as algebraic expressions
    320 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    321 
    322 Marco Mazzetti  
    323 Luigi Ciminiera 
    324 
    325 * http://dl.acm.org/citation.cfm?id=164360.164416
    326 * ~/opticks_refs/csg_tree_boundaries_as_expressions_p155-mazzetti.pdf
    327 
    328 Summary of the paper:
    329 
    330 * bbox obtained from a CSG tree depends on evaluation order !!, 
    331   as the bbox operation is not associative, 
    332 
    333 * solution is to rearrange the boolean expression tree into 
    334   a canonical form (UOI : union-of-intersections, aka sum-of-products) 
    335   which the paper states corresponds to the minimum bbox
    336 
    337 
    338 * upshot of this is that generally the bbox obtained will be overlarge
    339 
    339 
    340 * handling CSG difference requires defining an InnerBB 
    341   corresponding to the maximum aabb that is completely inside the shape, 
    342   then::
    343 
    344       BB(A - B) = BB(A) - InnerBB(B)
    345 
    346 
    347 */





