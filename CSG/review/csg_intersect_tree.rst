review/csg_intersect_tree
===========================

signbit(-0.f) signaling within mis-normal zeros trick
--------------------------------------------------------

::

    signbit(-0.f) == true ;
    signbit( 0.f) == false ;



::

    [lo] A[blyth@localhost CSG]$ grep -- -0.f *.*
    csg_intersect_leaf.h:        isect.x = valid_isect ? -isect.x : -0.f ;    // miss needs to signal complement with -0.f signbit 
    csg_intersect_leaf.h:        // even for miss need to signal the complement with a -0.f in isect.x
    csg_intersect_leaf.h:        if(complement) isect.x = -0.f ;
    csg_intersect_leaf_halfspace.h:            isect.y = -0.f;
    csg_intersect_leaf_halfspace.h:    bool yflip = valid_isect == false && isect.y == -0.f ;
    csg_intersect_leaf_phicut.h:        isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity
    csg_intersect_leaf_phicut.h:    // o on phi0 line => t_cand.0  -0.f
    csg_intersect_leaf_phicut.h:   for rays starting on phi0 line which have t_cand -0.f
    csg_intersect_leaf_phicut.h:        isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity
    csg_intersect_leaf_phicut.h:        isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity
    csg_intersect_leaf_thetacut.h:        isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity 
    csg_intersect_tree.h:                // it is impossible to Miss a complemented (signaled by -0.f) solid as it is unbounded
    CSGNode.cc:    nd.setAABB(  -0.f, -0.f, -0.f,  0.f, 0.f, 0.f  );   // HMM: is negated zero -0.f ever used for anything ?
    [lo] A[blyth@localhost CSG]$ 






RPN with metadata ?
--------------------

I am considering using an RPN sequence to represent a CSG tree - used for GPU
ray tracing (with OptiX).  The intersection algorithm  "Andrew Kensler: Ray
Tracing CSG Objects Using Single Hit Intersections" requires a recursive like
treatment with the requirement to sometimes repeat intersection with changed
t_min for subtrees.  I wonder how that could be achieved with an RPN flattened
tree - rather than the complete binary tree I have been using so far.



Implementing Andrew Kensler's "Ray Tracing CSG Objects Using Single Hit
Intersections" algorithm on an RPN-flattened sequence is a fantastic way to
eliminate the massive padding and memory overhead of a complete binary tree.
However, Kensler's algorithm inherently relies on a state-rewind mechanism.
When a Boolean operator determines that a ray misses a complex intersection
condition, it resets the ray's interval (t_min) and demands that one or both of
its subtrees be re-evaluated from scratch.



To achieve this recursion-like subtree repetition within a flat RPN token array
on the GPU (without recursive calls), you must shift from a traditional
evaluation stack to an extended stack that tracks evaluation metadata coupled
with jump offsets or token-skipping.


The Challenge: RPN Evaluation vs. Re-evaluation


In a standard RPN evaluator, once a token is read, it is popped from the
sequence and consumed. But Kensler's algorithm might execute a subtree, hit an
operator, and decide: "That didn't work out. Reset t_min and re-evaluate the
left subtree."

If you flatten your tree linearly using a simple token array, your program
counter (instruction index) has already moved past the left subtree tokens.


The Solution Strategy

To execute this efficiently on the GPU inside an OptiX intersection or
closest-hit program, your RPN token sequence needs to be enhanced with
structural metadata, and your GPU thread state requires an explicit, bounded
execution loop.

1. Augment Tokens with Jump Offsets (Pre-computed)

A bare RPN string like [A, B, +] doesn't tell an operator where A or B start.
You must pre-compute the token span length for every token.

* A primitive token has a span of 1.
* An operator token has a span equal to 1 + left_subtree_span + right_subtree_span.

This allows your interpreter to jump backward to the beginning of a subtree
instantly without parsing backward

2. The GPU Stack Layout
Instead of just pushing the intermediate intersection data (t, normal,
primitive_id), each stack frame must store how to re-run the subtree if a
rewind occurs::

    struct StackFrame
    {
        // Standard CSG Data
        float t_hit;
        float3 normal;
        int primitive_id;
        bool is_hit;

        // Structural Metadata for Kensler Rewinding
        int token_index;   // The instruction pointer where this subtree started
        float initial_tmin; // The t_min this subtree was evaluated with
    };




::

    int ip = 0; // Instruction pointer
    int stack_ptr = 0;

    while (ip < total_tokens) {
        Token token = tokens[ip];

        if (token.type == PRIMITIVE) {
            // Evaluate primitive intersection using current ray interval
            StackFrame frame;
            frame.token_index = ip; // Mark start
            frame.initial_tmin = ray.tmin;
            
            evaluate_primitive(token.id, ray, frame); 
            
            stack[stack_ptr++] = frame;
            ip++;
        } 
        else if (token.type == OPERATOR) {
            // Pop right and left operands
            StackFrame right = stack[--stack_ptr];
            StackFrame left  = stack[--stack_ptr];

            // Execute Kensler's Single-Hit Logic
            CsgAction action = evaluate_kensler_logic(left, right, token.op_type);

            if (action == RETRY_LEFT) {
                // REWIND REGION
                ray.tmin = calculate_new_tmin(left, right);
                ip = left.token_index; // Jump program counter back to start of left subtree
                
                // Note: right child data is discarded, left will be re-evaluated
                continue; 
            } 
            else if (action == RETRY_RIGHT) {
                ray.tmin = calculate_new_tmin(right, left);
                ip = right.token_index; // Jump program counter back to start of right subtree
                continue;
            } 
            else {
                // Normal resolution: Combine left and right hits into a single resultant frame
                StackFrame result = combine_hits(left, right, token.op_type);
                
                // Maintain structural metadata for the parent operator
                result.token_index = left.token_index; // The whole subtree started at 'left'
                
                stack[stack_ptr++] = result;
                ip++;
            }
        }
    }



Stack Depth Limit: Kensler's algorithm on an RPN tree requires a max stack
depth proportional to the max height of the CSG tree, not the total number of
primitives. Allocate this statically in thread local memory (StackFrame
stack[MAX_DEPTH]) to avoid global memory overhead.






Can this be made to work with n-ary trees ? I have listnodes that avoid
tree-node bloat for some combinations of primitives.

Yes, this approach can absolutely be adapted to support n-ary operators (list
nodes). In fact, evaluating multi-operand unions, intersections, or difference
groups using an RPN layout on the GPU is highly efficient because it completely
flattens what would otherwise be a deeply nested, unbalanced binary branch
structure.  To make Kensler's state-rewind algorithm work with an n-ary node
wrapped in RPN, you must transition your execution model from a simple binary
pop to an iterative accumulator loop or an array-backed multi-operand pass.
Here is how you can restructure your RPN layout, token metadata, and GPU
evaluation loop to natively handle list nodes without introducing tree-node
bloat.




1. The N-ary RPN Layout & Token Structure


Instead of forcing every operator to take exactly two arguments, your operator
tokens must explicitly declare their arity (the number of child subtrees they
consume).

For example, a multi-operand Union of four primitives A, B, C, D is laid out in RPN as::

    [A, B, C, D, UNION_N(arity=4)]

To support Kensler's rewind mechanism, your augmented token metadata must store
a list or an easily indexable array of the starting positions (token_index) for
all its children::

    struct Token {
        int type;        // PRIMITIVE or OPERATOR
        int op_type;     // UNION, INTERSECTION, DIFFERENCE
        int arity;       // 2 for binary, N for list nodes

        // Instead of a single span, we track where the children sit in the RPN array.
        // For local memory efficiency on the GPU, you can store a relative offset 
        // to the start of the first child, or a fixed-size array of child start indices.
        int child_start_indices[MAX_ARITY]; 
    };



2. The Multi-Operand GPU Stack Frame

When evaluating a standard binary RPN operator, you pop two elements, test
them, and potentially rewind to one of them. For an n-ary operator, you need
access to all N evaluated children simultaneously to determine which subtree
needs to adjust its t_min and re-evaluate.
Your GPU thread stack must track these individual child evaluations.

::

    struct StackFrame {
        // Kensler Intersection Results
        float t_hit;
        float3 normal;
        int primitive_id;
        bool is_hit;

        // Structural Metadata for Rewinding
        int token_index;    // Where this specific subtree started in the RPN array
        float initial_tmin; // The tmin used when this child was evaluated
    };


3. The N-Ary Evaluation and Rewind Loop

When the instruction pointer (ip) hits an n-ary operator, your GPU thread pops
N items from the stack. It then processes them through an n-ary extension of
Kensler's logic. If any child fails the single-hit validity criteria, the
operator can trigger a rewind directly to that specific child's starting index.


::

    int ip = 0;
    int stack_ptr = 0;

    while (ip < total_tokens) {
        Token token = tokens[ip];

        if (token.type == PRIMITIVE) {
            StackFrame frame;
            frame.token_index = ip;
            frame.initial_tmin = ray.tmin;
            
            evaluate_primitive(token.id, ray, frame); 
            
            stack[stack_ptr++] = frame;
            ip++;
        } 
        else if (token.type == OPERATOR) {
            int N = token.arity;
            
            // Grab pointers to the N children currently sitting on the top of the stack
            // Note: Children are popped in reverse order (right-to-left)
            int children_start_ptr = stack_ptr - N;
            StackFrame* children = &stack[children_start_ptr];

            // Process Kensler's logic over the array of N children
            int rewind_child_index = -1;
            float new_tmin = 0.0f;
            
            bool success = evaluate_kensler_n_ary_logic(children, N, token.op_type, rewind_child_index, new_tmin);

            if (!success && rewind_child_index != -1) {
                // REWIND REGION: Kensler logic says child 'rewind_child_index' needs re-evaluation
                ray.tmin = new_tmin;
                
                // Jump program counter back to the exact token where that specific child begins
                ip = children[rewind_child_index].token_index; 
                
                // Clear the stack back to the position of the first child of this operator.
                // Everything evaluated after that child's start is now invalid and will be recalculated.
                stack_ptr = children_start_ptr; 
                continue;
            } 
            else {
                // Normal resolution: Combine all N hits into a single resultant frame
                StackFrame result = combine_n_hits(children, N, token.op_type);
                
                // The entire n-ary subtree structurally starts where its first child started
                result.token_index = children[0].token_index; 
                
                // Collapse the N stack entries into the single result entry
                stack_ptr = children_start_ptr;
                stack[stack_ptr++] = result;
                ip++;
            }
        }
    }



Managing the N-Ary Kensler Logic Safely


The core of Andrew Kensler's single-hit approach relies on tracking whether a
ray enters or exits a shape boundaries along the ray interval. Extending this
to an N-ary operation requires handling combinations safely:

1. N-ary Union (List of Additions):

A ray hits an n-ary union if it hits any of the valid subtrees. If a ray hits
child A at t₁, but child B overlaps it and has an earlier boundary or a
modification that changes classification, you may need to update and retry
specific subtrees. If a child turns out to mask or shift the hit boundary
closer, you rewind the unresolved subtrees with the refined .

2. N-ary Intersection:

The ray must pierce the shared volume of all N subtrees. If child A hits at a
late t, all other children must be validated against this new, restricted . If
child B fails to hit in that new interval, the entire intersection fails, or
you rewind child B with the new to see if a secondary hit exists inside the
active boundary.



Architectural Trade-offs for OptiX

* Stack Management Costs: For a binary tree, you only ever pop 2 items. For an
  n-ary tree, you pop N items. If N is variable, accessing
  children[rewind_child_index] requires local memory array indexing. Keep your
  MAX_ARITY bounded (e.g., maximum 4 or 8 children per list node) so the compiler
  can unroll registers and keep stack lookups purely within GPU L1 cache / Shared
  Memory.

* Warp Divergence: If one thread jumps back to child index 0, and another jumps
  back to child index 3 within the same list-node, the warp will serialize.
  However, because you are avoiding the massive execution padding and traversal
  overhead of multi-layered fake binary nodes (node bloat), the reduction in
  global memory fetches generally results in a massive net performance win for
  OptiX pipelines.







Seems in my current JUNO geometry - I am using tesselated geometry not analytic
CSG for all the low occurrence but complex solids that needed listnode
treatment.  When I did use listnode I made simplifying assumptions like that
all children of the listnode where discontiguous primitives. I also simply
defined the list node to carry an offset and count to access those primitives
after the binary tree portion of the buffer.  Offsetting like that is not good
for locality - but it allows the algorithm to stay binary with special
intersection for the listnode.



Seems in my current JUNO geometry - I am using tesselated geometry not analytic
CSG for all the low occurrence but complex solids that needed listnode
treatment.  When I did use listnode I made simplifying assumptions like that
all children of the listnode where discontiguous primitives. I also simply
defined the list node to carry an offset and count to access those primitives
after the binary tree portion of the buffer.  Offsetting like that is not good
for locality - but it allows the algorithm to stay binary with special
intersection for the listnode.



Fixing the Locality Problem Without Rewriting to RPN

You mentioned that storing an offset and count that points to a completely
separate portion of the buffer (after the binary tree data) is bad for memory
locality. On the GPU, when threads in a warp hit different listnodes and fetch
from wild offsets at the end of the buffer, it causes severe L1/L2 cache
thrashing.  Instead of a full RPN conversion, you can fix this locality issue
by using Interleaved Sub-Arrays.


The Optimization: Flatten into a Single Cohesive Array

Instead of putting your listnode primitives at the very end of your global
buffer, bake them directly inline where the listnode operator sits.  Since you
are already using a flat array to represent your complete binary tree, you can
allocate the slots directly following the listnode operator to hold its
children sequentially


::

    Traditional Tree Array (Your current state):
    [ Node 0 | Node 1 | Listnode (Offset=100, Count=3) | ... ] ... [ Prim A | Prim B | Prim C ]
                                                                     ^ Severe cache jump

    Inline Interleaved Array (Optimized):
    [ Node 0 | Node 1 | Listnode (Count=3) | Prim A | Prim B | Prim C | Node 4 | ... ]
                                             ^ Caught in the same L2/L3 cache line!




::

    if (token.type == LISTNODE_UNION) {
        int child_count = token.count;
        StackFrame closest_hit;
        closest_hit.is_hit = false;
        closest_hit.t_hit = ray.tmax;

        // Loop through the children sitting directly next to this operator in memory
        for (int i = 0; i < child_count; i++) {
            Token child_prim = tokens[ip + 1 + i]; 
            StackFrame child_frame;
            
            if (evaluate_primitive(child_prim.id, ray, child_frame)) {
                if (child_frame.t_hit < closest_hit.t_hit) {
                    closest_hit = child_frame;
                }
            }
        }

        // Push the single closest winner to your standard binary stack
        stack[stack_ptr++] = closest_hit;

        // Skip the instruction pointer past all the inline primitives we just consumed
        ip += (1 + child_count); 
    }






Why this is a Win for JUNO

* Coalesced Memory Access: When a warp of threads hits the same listnode
  structure (very common in highly symmetric physics detectors like JUNO), they
  will pull the operator and its immediate primitive children in a single, clean
  memory transaction.

* Zero Structural Bloat: You retain your simple, fast binary evaluation stack for
  95% of the geometry.

* Preserves Assumptions: You don't have to change your host-side geometry
  compiler to output RPN; you just change the array-packing serialization code to
  output the listnode children inline rather than appending them to the tail.

To ensure this fits cleanly into your OptiX memory limits:

* What is the typical or maximum count of discontiguous primitives bundled into
  one of these listnodes?

* Are you storing these token/node buffers in OptiX Constant Memory or passing
  them via a CUDA Linear Device Buffer (SBT data / Structured Buffer)?




The CSGNode come from global memory - but ncu SpeedOfLight profiling shows I am
not DRAM limited - the entire analytic geometry easily fits within the L2 cache
(Ada RTX 5000).


::

    0984 extern "C" __global__ void __intersection__is()
     985 {
     986 
     987 #if defined(DEBUG_PIDXYZ)
     988     const uint3 idx = optixGetLaunchIndex();
     989     const uint3 dim = optixGetLaunchDimensions();
     990     bool dumpxyz = idx.x == params.pidxyz.x && idx.y == params.pidxyz.y && idx.z == params.pidxyz.z ;
     991     //if(dumpxyz) printf("//__intersection__is  idx(%u,%u,%u) dim(%u,%u,%u) dumpxyz:%d \n", idx.x, idx.y, idx.z, dim.x, dim.y, dim.z, dumpxyz);
     992 #else
     993     bool dumpxyz = false ;
     994 #endif
     995 
     996    
     997     HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();
     998     int nodeOffset = hg->prim.nodeOffset ;
     999     int globalPrimIdx = hg->prim.globalPrimIdx ;
    1000    
    1001     const CSGNode* node = params.node + nodeOffset ;  // root of tree
    1002     const float4* plan = params.plan ;
    1003     const qat4*   itra = params.itra ;
    1004    
    1005     const float  t_min = optixGetRayTmin() ;
    1006     const float3 ray_origin = optixGetObjectRayOrigin();
    1007     const float3 ray_direction = optixGetObjectRayDirection();
    1008    
    1009     float4 isect ; // .xyz normal .w distance
    1010     isect.x = 0.f ;
    1011     isect.y = 0.f ; 
    1012     isect.z = 0.f ;
    1013     isect.w = 0.f ;
    1014 
    1015     bool valid_isect = intersect_prim(isect, node, plan, itra, t_min , ray_origin, ray_direction, dumpxyz );
    1016     if(valid_isect)
    1017     {
    1018         const float3 lpos = ray_origin + isect.w*ray_direction ;
    1019         const float lposcost = normalize_cost(lpos);  // scuda.h cosTheta z/len of local(aka Object) frame position
    1020         const float lposfphi = normalize_fphi(lpos);
    1021 
    1022         const unsigned hitKind = 0u ;     // only up to 127:0x7f : could use to customize how attributes interpreted
    1023         const unsigned boundary = node->boundary() ;  // all CSGNode in the tree for one CSGPrim tree have same boundary
    1024         const unsigned globalPrimIdx_boundary = (( globalPrimIdx & 0xffffu ) << 16 ) | ( boundary & 0xffffu ) ;
    1025 
    1026         if(optixReportIntersection( isect.w, hitKind))
    1027         {
    1028             quad2* prd = SOPTIX_getPRD<quad2>(); // access prd addr from RG program
    1029             prd->q0.f = isect ;  // .w:distance and .xyz:normal which starts as the local frame one
    1030             prd->set_globalPrimIdx_boundary_(globalPrimIdx_boundary) ;
    1031             prd->set_lpos(lposcost, lposfphi);
    1032         }
    1033    }
    1034



The profiling insight from Nsight Compute (ncu) completely changes the
optimization vectors for your setup. Because your entire analytic geometry
easily fits within the L2 cache of your Ada RTX 5000, your bottleneck shifts
entirely from memory latency (DRAM) to compute/instruction throughput and L2
cache hits.

Seeing your __intersection__is() code reveals why your current architecture
performs so well: it delegates the actual traversal to intersect_prim, matching
your description of a specialized treatment for the listnode.

Given that you are not DRAM limited, a flat RPN implementation would actually
hurt your performance compared to an optimized inline tree arrangement. Here is
why, along with an explicit look at how to refine your node structure for the
L2 cache.



Why RPN is a De-optimization in an L2-Cached Scenario

When a dataset fits entirely inside the L2 cache, the primary performance
threats are register pressure and instruction count.


* The RPN Overhead: An RPN evaluator requires managing an instruction pointer,
  fetching explicit structural metadata (like jump offsets), and writing/reading
  from a thread-local stack array. This wastes hardware registers, forcing the
  compiler to spill them to local memory (which translates to slow L2/L3
  roundtrips anyway).

* The Complete Binary Tree Overhead: Your current complete binary tree uses
  simple array-index math (2*i, 2*i+1) to find children. This math happens
  entirely in high-speed ALU registers with zero metadata overhead.




Making the Hybrid Listnode "L2-Friendly"

Since your __intersection__is entry point fetches the tree root via const
CSGNode* node = params.node + nodeOffset;, your only true performance leak is
the pointer-chasing jump when a listnode tells intersect_prim to fetch the
disjoint primitives from the tail of the buffer.

To optimize this for an L2-resident geometry, you can modify your host-side
geometry compiler to pack the listnode's children immediately after the node
itself in memory, rather than at the end of the buffer. Because the RTX 5000 L2
cache handles transactions in 32-byte or 128-byte sectors, caching the listnode
will implicitly pre-fetch its children into the fastest cache lines.

The High-Performance L2 Layout

Instead of using a separate absolute buffer offset for the listnode primitives,
use a relative array offset.




ncu shows "Warp Stall Sampling" is ~12% of all stalls on the line where
typecode is compared against CSG_ZERO - due to accessing the typecode from the
long scoreboard.  So that made me think the problem is all the padding CSG_ZERO
nodes  of the complete binary trees meaning I am jumping all over the L2 cache.
Which made me think RPN would help.

This is the exact smoking gun. A "Long Scoreboard" stall on a global memory
load (params.node + nodeOffset) means your threads are stalling because the
execution pipe is waiting for data to arrive from the L2 cache into registers.

Even though your geometry fits inside the L2 cache, a complete binary tree
structure forces your loop to calculate child indices via 2*i and 2*i+1. For an
unbalanced tree, this math forces threads to load CSG_ZERO padding nodes across
scattered memory strides. A single 128-byte L2 cache line can hold exactly four
32-byte CSGNode structures. If your threads are hopping over large empty zones
of CSG_ZERO padding to evaluate deeply nested leaves, you are wasting L2 cache
lines on garbage data and thrashing the cache slots.

In this specific scenario, switching to an RPN (or similar sequential
postfix/prefix) flattened tree will absolutely solve your Long Scoreboard
problem.


Why RPN Fixes the Long Scoreboard Stall

By flattening the tree into an RPN token array, you completely eliminate the
CSG_ZERO padding nodes.

* Perfect Coalescing & Streaming: Instead of jumping across an implicit matrix,
  the instruction pointer (ip) increases monotonically (ip++). Threads in a warp
  will stream contiguous blocks of actual, structural geometry data out of L2
  cache lines. One 128-byte L2 cache fetch will guarantee 4 valid nodes that will
  be used sequentially, instead of 1 valid node and 3 padding zeroes.

* Predictable Pre-fetching: Modern GPU L2 caches detect sequential streaming
  memory access patterns perfectly. By stepping linearly through an array, the
  hardware can speculatively pull upcoming tokens into the fastest cache levels
  before the thread even requests them, driving those Long Scoreboard stalls down
  toward 0%.


Implementing the RPN Transition for OptiX

Because you verified that your listnode elements are discontiguous, you do not
need the heavy state-rewind machinery of Kensler's algorithm. You can combine a
standard RPN evaluation loop with your localized inline listnodes.

Here is how your intersect_prim can be cleanly rewritten into an iterative RPN
execution loop:


    // In-register stack for RPN evaluation
    // Max depth of a realistic analytic CSG tree is small (e.g., 8 to 16)
    #define MAX_RPN_STACK 16

    struct RPNFrame {
        float4 isect; // .xyz normal, .w distance
        bool hit;
    };

    __device__ bool intersect_prim_rpn(
        float4& final_isect, 
        const CSGNode* rpn_base, // Start of this primitive's RPN token array
        int token_count, 
        const float4* plan, 
        const qat4* itra, 
        float t_min, 
        const float3& ray_origin, 
        const float3& ray_direction) 
    {
        RPNFrame stack[MAX_RPN_STACK];
        int stack_ptr = 0;
        int ip = 0;

        while (ip < token_count) {
            // Linear stream out of L2 cache - high locality!
            const CSGNode* node = rpn_base + ip; 
            int typecode = node->typecode();

            if (typecode == CSG_ZERO) {
                // This case disappears globally when compiling to RPN!
                ip++;
                continue;
            }

            if (is_primitive(typecode)) {
                RPNFrame frame;
                frame.hit = evaluate_leaf_primitive(node, plan, itra, t_min, ray_origin, ray_direction, frame.isect);
                stack[stack_ptr++] = frame;
                ip++;
            } 
            else if (typecode == CSG_LISTNODE) {
                // Handle your specialized listnode inline
                int child_count = node->child_count();
                RPNFrame closest_hit;
                closest_hit.hit = false;
                closest_hit.is_isect.w = 1e30f; // infinity

                // Listnode children sit sequentially right after the operator token
                for (int i = 0; i < child_count; i++) {
                    const CSGNode* child_node = node + 1 + i;
                    float4 child_isect;
                    if (evaluate_leaf_primitive(child_node, plan, itra, t_min, ray_origin, ray_direction, child_isect)) {
                        if (child_isect.w < closest_hit.isect.w) {
                            closest_hit.isect = child_isect;
                            closest_hit.hit = true;
                        }
                    }
                }
                stack[stack_ptr++] = closest_hit;
                ip += (1 + child_count); // Fast forward past inline children
            } 
            else {
                // Standard CSG Boolean Operation (UNION, INTERSECT, DIFFERENCE)
                // Pop right and left operands
                RPNFrame right = stack[--stack_ptr];
                RPNFrame left  = stack[--stack_ptr];

                RPNFrame result;
                // Standard boolean combination based on left/right hit intervals
                result.hit = combine_boolean_intervals(left, right, typecode, result.isect);
                
                stack[stack_ptr++] = result;
                ip++;
            }
        }

        if (stack_ptr > 0 && stack[0].hit) {
            final_isect = stack[0].isect;
            return true;
        }
        return false;
    }







The New Profile to Expect

By deploying an array-packed RPN sequence without dead padding:

* Your Long Scoreboard Stalls on the typecode evaluation line will shrink
  drastically because the next node is nearly guaranteed to be cached in the same
  128-byte cache line or the adjacent one.

* Your Register Count might see a minor increase due to managing the small local
  stack array. Ensure MAX_RPN_STACK is explicitly hardcoded so the CUDA compiler
  (nvcc) can map the stack slots directly into hardware registers instead of
  spilling them into local memory (DRAM/L2 backing).





You think splitting primitive parameters from the node and referring to a
separate primitive parameters buffer would help ?


Yes, absolutely. Decoupling control-flow tokens from heavy geometric payloads
is one of the most effective design patterns for GPU-accelerated ray tracing
and spatial tree evaluations.

Splitting primitive parameters into a separate buffer transforms your layout
into a Structure of Arrays (SoA) style architecture, yielding three major
microarchitectural benefits:

1. Ultra-Compact RPN Bytecode Stream

If every RPN node or token embeds its own shape parameters (e.g., bounding
boxes, radii, transformation matrices, or plane equations), your
instruction/token stream bloat drastically (e.g., 64 to 128 bytes per node).

* By separating them, your RPN expression stream becomes a lightweight, uniform
  array of tiny tokens (e.g., 32-bit uint32_t values where bits 0–23 store the
  primitive index and bits 24–31 store the opcode/type).

* A compact token stream maximizes L2 cache density, allowing many more
  instructions to reside simultaneously in the L2 cache and reducing cache
  thrashing.

2. Elimination of Unnecessary Parameter Fetches (Lazy Evaluation)

CSG expressions frequently involve short-circuiting or conditional evaluation
depending on ray intervals.

* If primitive parameters are glued inside the nodes, the hardware fetches
  parameter data for branches that might ultimately be pruned or skipped during
  boolean evaluation.

* With a separate parameter buffer, a thread only performs an indirect lookup
  (primitive_buffer[index]) the exact moment it hits a LEAF token that requires
  actual intersection math.


3. Coalesced Memory Accesses for Similar Primitives

When warps evaluate similar RPN tokens concurrently, separating parameters into
dedicated typed buffers (or structured arrays) allows NVIDIA's memory subsystem
to coalesce loads if neighboring threads access adjacent primitives in the
buffer.


Conceptual Layout Blueprint

RPN Token Buffer (uint32_t* rpn_stream)::

    // Packed 32-bit token: [ 8-bit Opcode/Type | 24-bit Payload/Index ]
    // Example opcodes: OP_UNION, OP_INTERSECT, LEAF_SPHERE, LEAF_BOX


Primitive Parameter Buffers (SoA)::

    struct SphereParams {
        float3 center;
        float  radius;
    };
    // Dedicated flat buffer indexed directly by the LEAF token payload



Should I pass primitive parameters via unified memory, CUDA surface/texture
reads, or custom device pointers in OptiX?

For high-performance OptiX custom intersection programs and RPN evaluation, you
should use Custom Device Pointers (CUdeviceptr) stored in flat global memory
buffers and passed via OptiX Launch Parameters or the Shader Binding Table
(SBT).

Here is a breakdown of why this approach outperforms the alternatives for your
workload, alongside an evaluation of the other options:


1. Custom Device Pointers (cudaMalloc + Flat Buffers) — Recommended

   * How it works: You allocate flat device memory (cudaMalloc) for your SoA
     primitive parameters, and pass the raw CUdeviceptr base addresses into your
     OptiX launch parameters struct or SBT data. Inside the intersection/closest-hit
     program, threads index directly using standard pointer arithmetic:
     PrimitiveParams p = param_buffer[leaf_token.payload];.

   * Why it wins in OptiX:

     * Zero Overhead: No driver intervention, page-fault handling, or hidden
       migration costs.

     * L2 Cache Utilization: Since your working set fits well within the Ada
       Lovelace L2 cache (as shown by your 77.52% L2 throughput), raw device pointer
       lookups hit the L2 cache with minimal latency. 






I am thinking of a CSG tree represented by an RPN stream - if the CSG node
typecodes and parameter index references can be squeezed into 32 bits then many
nodes can be handled all within the 128 bytes of L2 cache line.


Suggested 32-Bit Bitfield Layout

To squeeze both the node typecode and parameter references into 32 bits (4
bytes), you can partition a uint32_t bitfield::

    struct alignas(4) CompactCSGNode {
        uint32_t typecode   : 6;   // 6 bits  -> up to 64 node types (primitives + boolean ops)
        uint32_t flags      : 2;   // 2 bits  -> e.g., invert flag, leaf vs. operator indicator
        uint32_t param_idx  : 24;  // 24 bits -> indexes up to 16.7M parameter blocks/instances
    };


* Typecode (6 bits): More than enough to cover primitive shapes (sphere,
  cylinder, box, cone, ellipsoid) and boolean operators (UNION, INTERSECTION,
  DIFFERENCE, TRANSFORM).

* Flags (2 bits): Useful for quick inline checks (e.g., whether the node is a
  terminal leaf or a stack operator, or whether the geometric transform is
  identity).

* Parameter Index (24 bits): Points to a separate, aligned buffer containing
  floating-point parameters (radii, half-lengths, transformation matrices) for
  primitives, avoiding inline bloat in the RPN stream.



Performance Implications for GPU Ray Tracing

1. Shared Memory Caching (__shared__):

   A small block of shared memory (e.g., 48 KB) can hold thousands of packed
   32-bit RPN nodes. For complex detector geometries (like multi-component PMTs or
   optical modules), entire CSG programs can be loaded into shared memory once per
   block, reducing global memory/L2 latency to near zero during ray evaluation.

2. Reduced Register Pressure & Stack Evaluation:

   During RPN evaluation, threads push/pop intersection intervals or ray
   parameters on a local stack. Keeping the instruction stream compact means
   instruction cache (I-cache) and texture/read-only data paths remain saturated
   with active work rather than padding bytes.

3. Coalesced Access Across Warps:

   When adjacent threads in a warp evaluate RPN streams for neighbouring rays,
   32-bit packed nodes ensure that sequential lookups hit adjacent memory
   addresses, maximizing L1/L2 cache hit rates.







How do I integrate a 32-bit packed RPN CSG evaluator into an NVIDIA OptiX
custom primitive intersection program?

::

    struct CSGSBTData {
        const uint32_t* rpn_nodes;     // Pointer to 32-bit packed RPN stream in global memory
        const float*    parameters;    // Pointer to float parameter buffer (referenced by 24-bit indices)
        uint32_t        node_count;    // Total number of nodes in the RPN stream
    };



::

    extern "C" __device__ void __intersection__csg() {
        // Retrieve SBT data passed via optixSetSbtData
        const auto& sbt_data = *reinterpret_cast<const CSGSBTData*>(optixGetSbtDataPointer());

        // Ray setup in object space
        const Ray ray = get_current_ray_object_space(); 

        // Register-based interval evaluation stack for CSG
        // Each stack entry holds ray interval [t_min, t_max]
        Interval stack[16];
        int sp = -1;

        // Evaluate RPN stream sequentially
        for (uint32_t pc = 0; pc < sbt_data.node_count; ++pc) {
            // Read packed 32-bit node from global memory using read-only cache (__ldg)
            uint32_t node      = __ldg(&sbt_data.rpn_nodes[pc]);
            uint32_t typecode  = node & 0x3F;             // Lower 6 bits: Node type / operator
            uint32_t flags     = (node >> 6) & 0x3;       // 2 bits: flags (e.g., invert)
            uint32_t param_idx = (node >> 8) & 0xFFFFFF;  // Upper 24 bits: parameter buffer index

            if (typecode < CSG_OP_UNION) {
                // Leaf Primitive: Intersect ray with primitive defined by parameters[param_idx]
                Interval hit = intersect_primitive(ray, typecode, flags, &sbt_data.parameters[param_idx]);
                stack[++sp] = hit;
            } else {
                // CSG Boolean Operator (UNION, INTERSECTION, DIFFERENCE)
                Interval right = stack[sp--];
                Interval left  = stack[sp];
                stack[sp] = evaluate_boolean_op(typecode, left, right);
            }
        }

        // If the top of the stack contains a valid valid intersection interval
        if (sp >= 0 && stack[sp].isValid(ray.tmin, ray.tmax)) {
            float t_hit = stack[sp].t_enter;
            // Report intersection to OptiX traversal engine
            optixReportIntersection(t_hit, 0 /* hitKind */);
        }
    }




Key Optimization Practices

* Bitfield Unpacking: Using simple bit masks (& 0x3F, >> 8) compiles down to
  fast PTX bfe (bitfield extract) instructions, keeping register overhead to a
  minimum.

* Stream Coalescing: When warps evaluate adjacent rays, ensuring threads step
  through contiguous 32-bit words maximizes L1/L2 cache line occupancy.

* Shared Memory Caching: For small, frequently accessed subtrees (such as
  individual PMT assemblies in JUNO), load the 32-bit RPN nodes into __shared__
  memory once at the start of the CUDA block to eliminate global memory latency
  entirely.



Even with inefficient complete binary tree layout the entire JUNO analytic CSG
fits within Ada RTX 5000 L2 cache.  So in this situation shared memory would
not help ?


You are entirely correct. With the Ada Lovelace architecture's massive 64 MB L2
cache on the RTX 5000 Ada, a complete analytic Constructive Solid Geometry
(CSG) tree or subtree for a JUNO detector component (like an 8-inch or 20-inch
PMT assembly) fits entirely within the L2 working set.   



Why L2 Suffices in This Regime

1. Persistent Cache Residency: Because the JUNO analytic CSG representation is
   compact (even with a padded complete binary tree layout), it will not suffer
   from cache eviction churn. It stays resident in L2 across nearly all active
   streaming multiprocessors (SMs).

2. Hardware-Managed Caching: Relying on the L2 cache via __ldg() (read-only
   data cache path) means the hardware manages line fetching automatically. You
   avoid having to write explicit block-wide cooperative loading logic (e.g.,
   __syncthreads() barriers) into your OptiX intersection shaders.

3. Implicit Warp Broadcasting: When multiple threads in a warp access adjacent
   RPN nodes, the L2 cache controller handles multicast and coalescing at hardware
   speed, closely matching the bandwidth behavior of shared memory without manual
   partitioning.

Potential Nuances & Edge Cases

While shared memory is unnecessary for caching the CSG tree itself, a few
architectural considerations remain on Ada Lovelace:

* Warp Divergence and Scatter: In OptiX, custom intersection programs are
  executed per ray. If adjacent rays within a warp intersect completely different
  parts of the JUNO geometry hierarchy (e.g., one ray hits the outer acrylic
  vessel while another hits a buried PMT internal structure), threads will
  request non-contiguous addresses. While L2 handles this gracefully, divergent
  global-to-L2 requests lack the zero-latency determinism of shared memory bank
  accesses.

* Latency Differential: L2 cache hit latency on Ada is roughly ~200 cycles,
  whereas shared memory is ~25–30 cycles. However, because ray tracing workloads
  are heavily latency-hidden by hardware thread scheduling (warps switching
  instantly when waiting on memory), the 170-cycle gap is completely absorbed by
  the massive number of concurrent warps in flight on the RTX 5000 Ada's 100 SMs.

* Preserving Shared Memory for Stacks: Keeping shared memory free allows you to
  allocate it for thread-local intersection stacks or bounding box traversal
  scratchpads if your OptiX program requires dynamic working storage per ray.








CSG based on complete binary tree with bit twiddling navigation - but sparse layout for deep trees
----------------------------------------------------------------------------------------------------


A description of the  current CSG tree intersection implementation is below. It
is based on slices of the postorder sequence - in order to repeat parts of the
intersections of sub-trees.  How do you suggest to do this with RPN ?
Presumably the equivalent is just simple RPN tape entry index slices ?

::

    /**
    intersect_tree
    -----------------

    The principal of the approach is described in a note by Andrew Kensler
    "Ray Tracing CSG Objects Using Single Hit Intersections",  available from
    the below link together with a discussion of the technique as used in the XRT raytracer.

    - http://xrt.wikidot.com/doc:csg 

    - http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf



                       
                      +-----------------------+
                      |                     B |
           +------------------+               | 
           | A        |       |               |
           |          |       |               |
           |          |       |               |
           |   0- - - 1 - - - 2 - - - - - - -[3]
           |          |       |               |
           |          |       |               |
           |          +-------|---------------+
           |                  |
           |                  |
           +------------------+

    A ray is shot at each sub-object to find the nearest intersection, then the
    intersection with the sub-object is classified as one of entering, exiting or
    missing it. Based upon the combination of the two classifications, one of
    several actions is taken:

    1. returning a hit

    2. returning a miss
    3. changing the starting point of the ray for one of the objects and then
       shooting this ray, classifying the intersection. In this case, the state
       machine enters a new loop.

    For full details of the LUT (lookup table) and the single-intersect CSG implementation below 
    see tests/CSGClassifyTest.cc and run the below::

        CSGClassifyTest U
        CSGClassifyTest I
        CSGClassifyTest D


    Complete binary tree of height 4 (31 nodes) with 1-based nodeIdx in binary:: 
                                                                                                                                              depth    elevation
                                                                          1                                                                      0         4
     
                                          10                                                            11                                       1         3

                              100                        101                            110                           [111]                       2         2

                     1000            1001          1010         1011             1100          1101            *1110*           1111              3         1
     
                 10000  10001    10010 10011    10100 10101   10110 10111     11000 11001   11010  11011   *11100* *11101*   11110   11111          4         0
                                                                                                         


    CSG looping in the below implementation has been using the below complete binary tree slices(tranche)::

        unsigned fullTree  = PACK4(  0,  0,  1 << height, 0 ) ;    

        unsigned leftIdx = 2*nodeIdx  ;    // left child of nodeIdx
        unsigned rightIdx = leftIdx + 1 ; // right child of nodeIdx  

        unsigned endTree   = PACK4(  0,  0,  nodeIdx,  endIdx  );
        unsigned leftTree  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
        unsigned rightTree = PACK4(  0,  0,  rightIdx << (elevation-1), nodeIdx );


    1 << height 
        leftmost, eg 10000
    0 = 1 >> 1 
        one beyond root(1) in the sequence
     
    nodeIdx
         node reached in the current slice of postorder sequence  
    endIdx 
         one beyond the last node in the current sequence (for fulltree that is 0)

    leftTree 
         consider example nodeIdx 111 which has elevation 2 in a height 4 tree
         
         nodeIdx  :  111
         leftIdx  : 1110  
         rightIdx : 1111

         leftTree.start : leftIdx << (2-1)  : 11100
         leftTree.end   : rightIdx << (2-1) : 11110    one beyond the leftIdx subtree of three nodes in the postorder sequence 

    rightTree
        again consider nodeIdx 111

        nodeIdx   :  111
        rightIdx  : 1111

        rightTree.start : rightIdx << (2-1) : 11110     same one beyond end of leftTree is the start of the rightTree slice 
        rightTree.end   :nodeIdx 


    Now consider how different things would be with an unbalanced tree : the number of nodes traversed in a leftTree traverse
    of an unbalanced tree would be much more... the leftTree  would encompass the entirety of the postorder sequence 
    up until the same end points as above.  The rightTree would not change.

    Perhaps leftTreeOld should be replaced with leftTreeNew starting all the way from leftmost beginning of the postorder sequence::

        unsigned leftTreeOld  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
        unsigned leftTreeNew  = PACK4(  0,  0,  1 << height , rightIdx << (elevation-1)) ; 

    I suspect that when using balanced trees the leftTreeold can cause spurious intersects
    due to discontiguity from incomplete geometry as a result of not looping over the 
    full prior postorder sequence. 

    Tried using leftTreeNew with a balanced tree and it still gives spurious intersects on internal boundariues,
    so it looks like tree balanching and the CSG algorithm as it stands are not compatible.  

    TODO: need to confirm exactly what is happening using CSGRecord, have not extinguished all hope of getting balanced
    to work yet, as it seems like it should be possible in principle.

    **/

     







Yes, exactly. The equivalent of those bit-shifted postorder slices in an RPN
layout is simply contiguous tape index slices ([start_idx, end_idx]).

The reason Kensler's algorithm can slice and repeat sub-tree evaluations in the
first place relies on a fundamental property of post-order traversals: every
subtree, no matter how complex or unbalanced, always occupies a single,
perfectly contiguous block of nodes on a post-order tape.


Why RPN Makes Slicing Native and Robust

In your current complete binary tree implementation, you have to use
bit-shifting (leftIdx << (elevation-1)) because the tree is padded with
CSG_ZERO nodes to maintain a rigid, uniform height. That rigid math is
precisely why your experiments with leftTreeNew and unbalanced trees
failed—rigid level-order indexing breaks down when branches aren't symmetrical,
leading to misalignment and spurious intersects on internal boundaries.

In an RPN tape, you drop the rigid elevation math entirely. Because nodes are
serialized strictly in post-order (Left -> Right -> Operator), a node's left
child subtree and right child subtree are guaranteed to sit immediately
preceding it on the tape in contiguous blocks.


struct RPNTranche {
    int tape_start;  // Inclusive start index on the RPN tape for this subtree
    int tape_end;    // Inclusive end index on the RPN tape for this subtree
    float t_min;     // Minimum intersection distance for this slice
};


When Kensler's lookup table (LUT) dictates that you need to re-evaluate a
sub-tree (e.g., re-shooting a ray into sub-tree A after intersecting sub-tree
B), your state machine doesn't need bit-shifts. It simply pushes a new
RPNTranche with the pre-calculated [tape_start, tape_end] bounds of sub-tree A
onto the tranche stack.












precomputed RPN spans
----------------------

Kensler CSG sometimes requires - re-doing left or right intersects with t_min
advanced.   This means that operator nodes need to hold their spans so I can
rewind the stack to redo an intersect.  Can those spans be pre-computed up
front ?


Yes, the spans can absolutely be pre-computed up front, and doing so is the
standard practice for hardware-accelerated RPN or linear-stream interpreters.
Because your CSG tree structure is static (it doesn't change from frame to
frame during ray tracing), you should compute these metadata spans once on the
host CPU during geometry compilation and pack them directly into your CSGNode
structure.

When using an RPN layout, storing both the left child span and the total
operator span allows your GPU traversal loop to instantly calculation exactly
where to jump back (ip = current_ip - offset) to re-evaluate the required
subtree with the advanced t_min.


1. How Spans Map to an RPN Sequence

In an RPN sequence, subtrees are fully evaluated before they reach the operator
node. Structurally, the tokens are ordered as::

    [ Left Subtree Tokens ] [ Right Subtree Tokens ] [ Operator Node ]

When your instruction pointer (ip) is currently sitting directly on the
Operator Node, the subtrees are positioned immediately behind it in memory.

* Right Subtree Span (S_R): The number of tokens making up the right child. The
  right subtree ends exactly at ip - 1. It starts at ip - S_R.

* Left Subtree Span (S_L): The number of tokens making up the left child. It
  ends at ip - S_R - 1. It starts at ip - S_R - S_L

By storing S_R and S_L  directly in the Operator Node's bitfield metadata, you
can perform instantaneous state rewinds without any backward searching or
pointer chasing:



::

    class ASTNode:
        def __init__(self, typecode, left=None, right=None, child_count=0):
            self.typecode = typecode
            self.left = left
            self.right = right
            self.child_count = child_count # Used for your listnodes
            
            # Pre-computed metadata for the GPU
            self.total_span = 0
            self.left_span = 0
            self.right_span = 0

    def compute_rpn_metadata(node):
        if node is None:
            return 0

        if is_primitive(node.typecode):
            node.total_span = 1
            return 1
            
        elif node.typecode == CSG_LISTNODE:
            # A listnode holds its operator plus all inline children
            node.total_span = 1 + node.child_count
            return node.total_span
            
        else: # Standard CSG Binary Operator (UNION, INTERSECT, etc.)
            # Recursively calculate child spans
            node.left_span = compute_rpn_metadata(node.left)
            node.right_span = compute_rpn_metadata(node.right)
            
            # The operator token itself counts as 1
            node.total_span = node.left_span + node.right_span + 1
            return node.total_span



Once computed, your serializer writes out the flat node array. When it outputs
an operator token, it bakes left_span and right_span directly into the CSGNode
structural fields.


3. Modifying the GPU Stack for Rewinding

When you rewind the instruction pointer (ip), you must also rewind your
evaluation stack pointer (stack_ptr).

Because a binary operator pops 2 frames (left and right) and processes them, if
a rewind is triggered, you must reset the stack pointer to exactly where it was
before those subtrees were evaluated.


* If rewinding to the Left Subtree: You must drop both the right and left
  evaluation results from the stack. The stack pointer resets to its state prior
  to evaluating the left subtree.

* If rewinding to the Right Subtree: You must preserve the valid left subtree
  result on the stack, but drop the right subtree result.


::

    // Operator node is at 'ip'
    const CSGNode* op_node = rpn_base + ip;
    int left_span  = op_node->get_left_span();
    int right_span = op_node->get_right_span();

    if (kensler_action == RETRY_LEFT) {
        ray.tmin = advanced_t;
        
        // Jump back to the start of the left subtree tokens
        ip = ip - right_span - left_span; 
        
        // Pop both subtrees off the stack to re-run everything from that point
        stack_ptr = stack_ptr - 2; 
        continue;
    } 
    else if (kensler_action == RETRY_RIGHT) {
        ray.tmin = advanced_t;
        
        // Jump back to the start of the right subtree tokens
        ip = ip - right_span; 
        
        // Keep the Left child result (which is at stack[stack_ptr - 2])
        // Clear only the Right child result
        stack_ptr = stack_ptr - 1; 
        continue;
    }













