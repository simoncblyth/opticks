#pragma once


static __device__
void intersect_boolean_only_first( const uint4& prim, const uint4& identity )
{
    unsigned a_partIdx = prim.x + 1 ;  


    float tA_min = propagate_epsilon ;  
    float4 tt = make_float4(0.f,0.f,1.f, tA_min);

    //IntersectionState_t a_state = intersect_part( a_partIdx , tA_min, tt ) ;
    intersect_part( a_partIdx , tA_min, tt ) ;

    IntersectionState_t a_state = tt.w > tA_min ? 
                        ( (tt.x * ray.direction.x + tt.y * ray.direction.y + tt.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                              Miss
                              ; 

    if(a_state != Miss)
    {
        if(rtPotentialIntersection(tt.w))
        {
            shading_normal.x = geometric_normal.x = tt.x ;
            shading_normal.y = geometric_normal.y = tt.y ;
            shading_normal.z = geometric_normal.z = tt.z ;

            instanceIdentity = identity ;
            //instanceIdentity.x = dot(a_normal, ray.direction) < 0.f ? 1 : 2 ;

            rtReportIntersection(0);
        }
    }
}


#define CSG_STACK_SIZE 4
#define TRANCHE_STACK_SIZE 4

#define POSTORDER(i) ((postorder & (0xFull << (i)*4 )) >> (i)*4 ) 


static __device__
void intersect_csg( const uint4& prim, const uint4& identity )
{
    // see opticks/dev/csg/node.py:Node.postOrderSequence
    // sequence of levelorder indices in postorder, which has tree meaning 
    const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;

    int ierr = 0 ;  
    bool abort_ = false ; 

    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ;
    unsigned primIdx_   = prim.z ; 

    unsigned fullHeight = __ffs(numParts + 1) - 2 ;   // assumes perfect binary tree node count       2^(h+1) - 1 
    unsigned height = fullHeight - 1;                 // exclude leaves, triplet has height 0

    unsigned long long postorder = postorder_sequence[height] ; 
    unsigned numInternalNodes = (0x1 << (1+height)) - 1 ;

    // the tranche indices pick ranges of the postorder sequence
    // 0-based indices into postorder sequence


    float tmin = 0.f ;  

    // allocate stacks

    float4 _lhs[CSG_STACK_SIZE] ; 
    int lhs = -1 ; 

    float4 _rhs[CSG_STACK_SIZE] ; 
    int rhs = -1 ; 

    float4 _tmin ;  // TRANCHE_STACK_SIZE is 4 
    uint4  _tranche ; 
    int tranche = -1 ;

    float4 miss   = make_float4(0.f,0.f,1.f,0.f);
    float4 result = make_float4(0.f,0.f,1.f,0.f) ; 

    tranche++ ;  // push 0-based postorder indices
    setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (0 & 0xffff) )  ; // postorder end, begin
    setByIndex(_tmin,    tranche,  tmin ); 

    while (tranche >= 0)
    {
         float   tmin = getByIndex(_tmin, tranche);
         unsigned tmp = getByIndex(_tranche, tranche );
         unsigned begin = tmp & 0xffff ;
         unsigned end   = tmp >> 16 ;
         tranche-- ;                // pop, -1 means empty stack

         //rtPrintf("intersect_csg: begin %u end %u \n", begin, end );  // 0, 3

         for(unsigned i=begin ; i < end ; i++)
         {
             // XXidx are 1-based levelorder perfect tree indices
             unsigned nodeIdx = (postorder & (0xFull << i*4 )) >> i*4 ;   
             unsigned leftIdx = nodeIdx*2 ; 
             unsigned rightIdx = nodeIdx*2 + 1; 
             int depth = 32 - __clz(nodeIdx)-1 ;  
             unsigned subNodes = (0x1 << (1+height-depth)) - 1 ; // subtree nodes  
             unsigned halfNodes = (subNodes - 1)/2 ;             // nodes to left or right of subtree

             bool bileaf = leftIdx > numInternalNodes ; 

             //if(i>1)
             //rtPrintf("intersect_csg: i %u nodeIdx %u leftIdx %u rightIdx %u numInternalNodes %u bileaf %d  \n", i, nodeIdx, leftIdx, rightIdx, numInternalNodes, bileaf );  
             // intersect_csg: i 0 nodeIdx 2 leftIdx 4 rightIdx 5 numInternalNodes 3 bileaf 1  
             // intersect_csg: i 1 nodeIdx 3 leftIdx 6 rightIdx 7 numInternalNodes 3 bileaf 1 
             // intersect_csg: i 2 nodeIdx 1 leftIdx 2 rightIdx 3 numInternalNodes 3 bileaf 0 

             //if(i > 1)
             //rtPrintf("intersect_csg: i %u nodeIdx %u depth %u height %u subNodes %u halfNodes %u \n", i, nodeIdx, depth, height, subNodes, numInternalNodes, halfNodes ); 
             // i 0 nodeIdx 2 depth 1 height 1 subNodes 1 halfNodes 0 
             // i 1 nodeIdx 3 depth 1 height 1 subNodes 1 halfNodes 0 
             // i 2 nodeIdx 1 depth 0 height 1 subNodes 3 halfNodes 1
 
             quad q1 ; 
             q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];
             OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

             //if(i>1)
             //rtPrintf("intersect_csg: i %u nodeIdx %u operation %u \n", i, nodeIdx, (int)operation );
             //  intersect_csg: i 0 nodeIdx 2 operation 2              UNION:0/INTERSECTION:1/DIFFERENCE:2/PRIMITIVE:3
             //  intersect_csg: i 1 nodeIdx 3 operation 2         
             //  intersect_csg: i 2 nodeIdx 1 operation 0 


             float4 left  = make_float4(0.f,0.f,1.f,0.f);
             float4 right = make_float4(0.f,0.f,1.f,0.f);

             float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
             float tB_min = ray.tmin ;

             int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 
             bool reiterate = false ; 


             int loop(-1) ;  
             while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && loop < 4 )
             {
                loop++ ; 

                if(ctrl & CTRL_LOOP_A)
                {
                    if(bileaf) // left leaf node 
                    {
                         intersect_part( partOffset+leftIdx-1 , tA_min, left  ) ;
                         //if(loop > 0)
                         //rtPrintf("intersect_csg(bileaf) loop %u leftIdx %u tA_min %10.3f   (%10.2f, %10.2f,%10.2f,%10.2f) a_state %d \n",loop, leftIdx,tA_min, left.x, left.y, left.z, left.w, a_state );  
                         //intersect_csg(bileaf) leftIdx 4 tA_min      0.100   (      0.15,      -0.30,     -0.94,    387.15) a_state 0         Enter/Exit/Miss  0/1/2
                         //intersect_csg(bileaf) leftIdx 4 tA_min      0.100   (      0.00,       0.00,      1.00,      0.10) a_state 2 
                         //intersect_csg(bileaf) leftIdx 4 tA_min      0.100   (      0.00,       0.00,      1.00,      0.10) a_state 2 
                         //intersect_csg(bileaf) leftIdx 4 tA_min      0.100   (      0.14,       0.48,     -0.87,    334.24) a_state 0 
                    }
                    else                             // operation node
                    {
                         if(lhs >= 0)
                         {
                             left = _lhs[lhs] ;  
                             //rtPrintf("intersect_csg(non-bileaf-op) nodeIdx %u pop lhs %d (%10.2f, %10.2f,%10.2f,%10.2f) \n", nodeIdx,lhs,left.x,left.y,left.z,left.w);
                             //intersect_csg(non-bileaf-op) nodeIdx 1 pop lhs 0 (      0.00,       0.00,      1.00,      0.00) 
                             //intersect_csg(non-bileaf-op) nodeIdx 1 pop lhs 0 (      0.22,       0.51,      0.83,    296.03) 
                             //intersect_csg(non-bileaf-op) nodeIdx 1 pop lhs 0 (      1.00,       0.00,      0.00,    184.11) 
                             //intersect_csg(non-bileaf-op) nodeIdx 1 pop lhs 0 (      0.00,       1.00,      0.00,    249.89) 

                             lhs-- ;          // pop
                         }
                         else
                         {
                             ierr |= ERROR_LHS_POP_EMPTY ; 
                             left = miss ; 
                         } 
                    }
                } // CTRL_LOOP_A

                if(ctrl & CTRL_LOOP_B)
                {
                    if(bileaf)  // right leaf node
                    {
                         intersect_part( partOffset+rightIdx-1 , tB_min, right  ) ;
                         //if(loop > 0)
                         //rtPrintf("intersect_csg(bileaf) loop %u rightIdx %u tB_min %10.3f   (%10.2f, %10.2f,%10.2f,%10.2f) b_state %d \n",loop, rightIdx,tB_min, right.x, right.y, right.z, right.w, b_state );  
                         //intersect_csg(bileaf) rightIdx 5 tB_min      0.100   (     -0.65,      -0.51,     -0.56,    235.63) b_state 0 
                         //intersect_csg(bileaf) rightIdx 5 tB_min      0.100   (     -0.85,      -0.51,     -0.14,    208.01) b_state 0 
                         //intersect_csg(bileaf) rightIdx 5 tB_min      0.100   (     -0.60,      -0.05,     -0.80,    273.04) b_state 0 
                    }
                    else        // operation node
                    {
                         if(rhs >= 0)
                         {
                             right = _rhs[rhs] ;  

                            //rtPrintf("intersect_csg(non-bileaf-op) nodeIdx %u pop rhs %d (%10.2f, %10.2f,%10.2f,%10.2f) \n", nodeIdx,rhs,right.x,right.y,right.z,right.w);
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (     -0.34,       0.08,      0.94,     12.93) 
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (     -0.26,      -0.05,     -0.96,    368.14) 
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (     -0.29,      -0.35,      0.89,     21.71) 
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (      0.00,       0.00,      1.00,      0.00) 
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (      0.00,       0.00,      1.00,      0.00) 
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (      0.78,       0.40,     -0.49,    205.49) 
                            //intersect_csg(non-bileaf-op) nodeIdx 1 pop rhs 0 (      0.28,       0.37,     -0.88,    455.91) 

                             rhs-- ;          // pop
                         }
                         else
                         {
                             ierr |= ERROR_RHS_POP_EMPTY ; 
                             right = miss ; 
                         } 
                    }
                } // CTRL_LOOP_B
 


                IntersectionState_t a_state = left.w > tA_min ? 
                        ( (left.x * ray.direction.x + left.y * ray.direction.y + left.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

                IntersectionState_t b_state = right.w > tB_min ? 
                        ( (right.x * ray.direction.x + right.y * ray.direction.y + right.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 


                int actions = boolean_actions( operation , a_state, b_state );
                int act = boolean_decision( actions, left.w <= right.w );
                ctrl = boolean_ctrl( act );

                //if(nodeIdx == 1 && ctrl != 4)
                rtPrintf("intersect_csg: nodeIdx %u operation %u a_state %u b_state %u actions %8x  %10.2f %10.3f  act %8x ctrl %u \n", nodeIdx, operation,a_state,b_state,actions, left.w, right.w, act, ctrl ); 

                //  always getting Miss, Miss -> CTRL_RETURN_MISS
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1      115.36    455.906  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1        0.00    359.694  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1      330.20      0.000  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1      276.04      0.000  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1      300.83      0.000  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1      353.44      0.000  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1        0.00    243.569  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1      417.77     17.773  act        1 ctrl 4 
                //intersect_csg: nodeIdx 1 operation 0 a_state 2 b_state 2 actions        1        0.00    236.591  act        1 ctrl 4 


                //intersect_csg: nodeIdx 2 operation 2 a_state 0 b_state 0 actions      202      254.85    203.699  act      200 ctrl 2 
                //intersect_csg: nodeIdx 2 operation 2 a_state 0 b_state 0 actions      202      255.32    204.043  act      200 ctrl 2 
                //intersect_csg: nodeIdx 2 operation 2 a_state 0 b_state 0 actions      202      241.18    170.451  act      200 ctrl 2 
                //intersect_csg: nodeIdx 2 operation 2 a_state 0 b_state 0 actions      202      212.28    152.776  act      200 ctrl 2 
                //intersect_csg: nodeIdx 2 operation 2 a_state 2 b_state 0 actions        1        0.10    245.410  act        1 ctrl 4 
                //intersect_csg: nodeIdx 2 operation 2 a_state 0 b_state 0 actions      202      324.69    270.207  act      200 ctrl 2 
/*
dump_ctrl_enum
    0    1                  CTRL_LOOP_A 
    1    2                  CTRL_LOOP_B 
    2    4             CTRL_RETURN_MISS 
    3    8                CTRL_RETURN_A 
    4   10                CTRL_RETURN_B 
    5   20           CTRL_RETURN_FLIP_B 
    6   40                   CTRL_ERROR 
*/

                if(ctrl == CTRL_LOOP_A) 
                {
                    tA_min = left.w  ;  // epsilon ? 

                    if(!bileaf)   // left is not leaf
                    {
                         if(rhs+1 >= CSG_STACK_SIZE)
                         {
                             ierr |= ERROR_RHS_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         rhs++ ;   // push other side, as just popped it while reiterating this side
                         _rhs[rhs] = right ;    

                         if(tranche+2 >= TRANCHE_STACK_SIZE)
                         { 
                             ierr |= ERROR_LHS_TRANCHE_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         tranche++ ;  // push, from here on up : i -> numInternalNodes
                         setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (i & 0xffff) )  ;  
                         setByIndex(_tmin,    tranche,  tmin );

                         tranche++ ;  // push, left subtree  :  i - 2*halfNodes -> i - halfNodes
                         setByIndex(_tranche, tranche, ((i-halfNodes & 0xffff) << 16) | ((i-2*halfNodes) & 0xffff) )  ;
                         setByIndex(_tmin,    tranche,  tA_min );

                         reiterate = true ; 
                    } 
                } 
                else if(ctrl == CTRL_LOOP_B) 
                {
                    tB_min = right.w ;   // epsilon ?

                    if(!bileaf)   // left is not leaf
                    {
                         if(lhs+1 >= CSG_STACK_SIZE)
                         {
                             ierr |= ERROR_LHS_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         lhs++ ;   // push other side
                         _lhs[lhs] = left ;    


                         if(tranche+2 >= TRANCHE_STACK_SIZE)
                         { 
                             ierr |= ERROR_RHS_TRANCHE_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         tranche++ ;  // push, from here on up : i -> numInternalNodes
                         setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (i & 0xffff) )  ;  
                         setByIndex(_tmin,    tranche,  tmin );

                         tranche++ ;  // push, right subtree :  i - halfNodes -> i
                         setByIndex(_tranche, tranche, ((i & 0xffff) << 16) | ((i-halfNodes) & 0xffff) )  ;
                         setByIndex(_tmin,    tranche,  tB_min );

                         reiterate = true ; 
                    } 
                }
                if(reiterate) break ;
             }  // end while : ctrl loop


             if(reiterate || abort_) break ;  
             // reiteration needs to get back to tranche loop for subtree traversal 
             // without "return"ing anything


             if( ctrl == CTRL_RETURN_MISS )
             {
                 result = miss ; 
             }
             else if(ctrl == CTRL_RETURN_A) 
             {
                 result = left ; 
             } 
             else if( ctrl == CTRL_RETURN_B )
             {
                 result = right ; 
             }
             else if( ctrl == CTRL_RETURN_FLIP_B )
             {
                 result.x = -right.x ; 
                 result.y = -right.y ; 
                 result.z = -right.z ; 
                 result.w =  right.w ; 
             }
             else
             {
                  ierr |= ERROR_BAD_CTRL ; 
             }   
         
             if(nodeIdx % 2 == 0) // even 1-based nodeIdx is left
             {
                 lhs++ ;   // push
                 _lhs[lhs] = result ;    
                 //rtPrintf("intersect_csg : nodeIdx %d ctrl %4x push lhs %d  (%10.3f %10.3f %10.3f %10.3f)\n" , nodeIdx, ctrl, lhs, result.x, result.y, result.z, result.w );
                 //intersect_csg : nodeIdx 2 ctrl   10 push lhs 0  (    -0.924      0.372     -0.089    115.358)
                 //intersect_csg : nodeIdx 2 ctrl   10 push lhs 0  (     0.212      0.475      0.854    353.440)
                 //intersect_csg : nodeIdx 2 ctrl    8 push lhs 0  (     0.000      0.000      1.000    543.476)
                 //intersect_csg : nodeIdx 2 ctrl    4 push lhs 0  (     0.000      0.000      1.000      0.000)
                 //intersect_csg : nodeIdx 2 ctrl   10 push lhs 0  (    -0.291     -0.348      0.891    421.710)
                 //intersect_csg : nodeIdx 2 ctrl   10 push lhs 0  (     0.130     -0.115      0.985    383.785)
             }
             else
             {
                 rhs++ ;   // push
                 _rhs[rhs] = result ;    
                 //rtPrintf("intersect_csg : nodeIdx %d ctrl %4x push rhs %d (%10.3f %10.3f %10.3f %10.3f)\n" , nodeIdx, ctrl, rhs, result.x, result.y, result.z, result.w );
                 //intersect_csg : nodeIdx 3 ctrl    4 push rhs 0 (     0.000      0.000      1.000      0.000)
                 //intersect_csg : nodeIdx 3 ctrl    4 push rhs 0 (     0.000      0.000      1.000      0.000)
                 //intersect_csg : nodeIdx 3 ctrl   10 push rhs 0 (     0.782      0.162     -0.602    243.569)
                 //intersect_csg : nodeIdx 3 ctrl    4 push rhs 0 (     0.000      0.000      1.000      0.000)
                 //intersect_csg : nodeIdx 3 ctrl   10 push rhs 0 (     0.092      0.841     -0.533    231.761)
                 //intersect_csg : nodeIdx 3 ctrl    4 push rhs 0 (     0.000      0.000      1.000      0.000)
                 //intersect_csg : nodeIdx 3 ctrl   10 push rhs 0 (    -0.083     -0.537     -0.840    312.305)
                 //intersect_csg : nodeIdx 3 ctrl   10 push rhs 0 (     0.980     -0.115     -0.165    143.428)
             }

         }  // end for : node traversal within tranche
    }       // end while : tranche

    

    if(lhs != -1) ierr |= ERROR_LHS_END_NONEMPTY ;  
    if(rhs != 0)  ierr |= ERROR_RHS_END_EMPTY  ; 

    if(rhs == 0 && ierr == 0)
    {
         result = _rhs[rhs] ;  
         rhs-- ;  // pop
         if(rtPotentialIntersection( result.w ))
         {
              shading_normal = geometric_normal = make_float3(result.x, result.y, result.z) ;
              instanceIdentity = identity ;
              rtReportIntersection(0);
         }
    } 

    //rtPrintf("intersect_csg partOffset %u numParts %u numInternalNodes %u primIdx_ %u height %u postorder %llx ierr %x \n", partOffset, numParts, numInternalNodes, primIdx_, height, postorder, ierr );
    rtPrintf("intersect_csg primIdx_ %u ierr %4x  (%10.3f %10.3f %10.3f %10.3f)   \n", primIdx_, ierr,  result.x, result.y, result.z, result.w  );

}   // intersect_csg


static __device__
void intersect_boolean_triplet( const uint4& prim, const uint4& identity )
{
    // NB LIMITED TO SINGLE BOOLEAN OPERATION APPLIED TO TWO BASIS SOLIDS, ie triplet trees

    // primFlags only available for root of tree,
    // operate from partBuffer for other nodes

    unsigned partOffset = prim.x ; 
    //unsigned primIdx_   = prim.z ; 

    unsigned n_partIdx = partOffset ;    
    unsigned a_partIdx = partOffset + 1 ;   // SIMPLIFYING TRIPLET ASSUMPTION
    unsigned b_partIdx = partOffset + 2 ;  

    quad q1 ; 
    q1.f = partBuffer[4*n_partIdx+1];
    OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

    //rtPrintf("intersect_boolean primIdx_:%u n:%u a:%u b:%u operation:%u \n", primIdx_, n_partIdx, a_partIdx, b_partIdx, operation );

    float4 left  = make_float4(0.f,0.f,1.f,0.f);
    float4 right = make_float4(0.f,0.f,1.f,0.f);

    float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
    float tB_min = ray.tmin ;

    int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 


    int count(0) ;  
    while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && count < 4 )
    {
        count++ ; 

        if(ctrl & CTRL_LOOP_A) intersect_part( a_partIdx , tA_min, left  ) ;
        if(ctrl & CTRL_LOOP_B) intersect_part( b_partIdx , tB_min, right ) ;

        IntersectionState_t a_state = left.w > tA_min ? 
                        ( (left.x * ray.direction.x + left.y * ray.direction.y + left.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

        IntersectionState_t b_state = right.w > tB_min ? 
                        ( (right.x * ray.direction.x + right.y * ray.direction.y + right.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

        int actions = boolean_actions( operation , a_state, b_state );
        int act = boolean_decision( actions, left.w <= right.w );
        ctrl = boolean_ctrl( act );

        if(     ctrl == CTRL_LOOP_A) tA_min = left.w  ;  // no epsilon ? 
        else if(ctrl == CTRL_LOOP_B) tB_min = right.w ; 
    } 


    // hmm below passing to OptiX should probably be done in caller ?
    if( ctrl & (CTRL_RETURN_A | CTRL_RETURN_B | CTRL_RETURN_FLIP_B  ))
    {
        if(rtPotentialIntersection( ctrl == CTRL_RETURN_A ? left.w : right.w ))
        {
            shading_normal = geometric_normal = ctrl == CTRL_RETURN_A ? 
                                                                           make_float3(left.x, left.y, left.z)
                                                                      :
                                                                          ( ctrl == CTRL_RETURN_FLIP_B ? -make_float3(right.x, right.y, right.z) : make_float3(right.x, right.y, right.z) )
                                                                      ;
            instanceIdentity = identity ;
            rtReportIntersection(0);
        }
    } 

}



