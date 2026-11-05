ncu_profiling_points_at_csg_impl_as_bottleneck
===============================================


ncu cxs_min.sh SpeedOfLight : bottleneck from L2 latency
-----------------------------------------------------------

The Opticks simulation kernel with the JUNO geometry has a bounce loop that
does trace then propagate. Manual atomicAdd clock measuments show the  trace
stage takes >100x times the propagate stage doing the simulation.    The ncu
results point to the analytic CSG implementation which is invoked for the
mostly custom analytic geometry being the bottleneck and explaining slow trace
stage.    Applying SER gave me about factor of 2x. My BNL colleagues who report
16x from SER with their geometry report 16x from SER.  They do not detail their
geometry.  Do you think their geometry being less of a bottleneck could explain
the huge difference in the SER improvement ? Or does it do the otherway.   What
can I measure to pin this down ? 


 The CSG is memory bound. For every CSG intersect the tree of CSGNode must be traversed


[3129193] CSGOptiXSMTest@127.0.0.1

  optixLaunch (1, 1, 1)x(512, 1, 1), Context 1, Device 0, CC 8.9

    Section: GPU Speed Of Light Throughput

    ----------------------- ----------- --------------

    Metric Name             Metric Unit   Metric Value

    ----------------------- ----------- --------------

    DRAM Frequency                  Ghz           8.51

    SM Frequency                    Ghz           1.15

    Elapsed Cycles                cycle    272,683,289

    Memory Throughput                 %          77.52

    DRAM Throughput                   %           1.08

    Duration                         ms         237.07

    L1/TEX Cache Throughput           %          57.03

    L2 Cache Throughput               %          77.52

    SM Active Cycles              cycle 235,134,637.18

    Compute (SM) Throughput           %          37.76

    ----------------------- ----------- -------------- 



The Nsight Compute profile metrics reveal a very specific and telling
signature: the bottleneck is not DRAM bandwidth, but L2 Cache Latency and
non-coalesced access.


1. Decoding the Nsight Compute Profile

   * DRAM Throughput (1.08%) vs. L2 Cache Throughput (77.52%): Your entire
     CSGNode tree is fitting comfortably inside the RTX 5000 Ada’s massive 64 MB L2
     cache. Because it fits in L2, data is rarely fetched from system DRAM.

   * Compute Throughput (37.76%) & Low SM Clock (1.15 GHz): Despite fitting in
     L2, compute utilization is low and the clock frequency has dropped. This
     happens because tree traversal is pointer-chasing or index-jumping. Threads
     within a warp are evaluating different branches of arbitrary CSG trees, causing
     random cache line lookups, high memory latency stalls, and idle execution
     pipelines.


2. Why CSG Tree Traversal Behaves This Way

   In analytic CSG geometry (like JUNO’s nested PMTs and acrylic structures), an
   intersection query requires walking a binary tree of boolean operations (CSGNode).

   1. Divergent Pointer Chasing: Unlike triangle meshes where memory access follows
      predictable spatial bounding box layouts, evaluating a CSG tree involves
      reading node types, left/right child indices, and primitive parameters
      dynamically based on intermediate intersection results.

   2. Warp-Level Chaos: If thread 0 in a warp is evaluating the outer rim of a PMT
      bulb while thread 1 is evaluating the internal dynode structure, their CSGNode
      memory requests bounce around different regions of the L2 cache line,
      destroying spatial locality.



3. How This Connects to Your SER Results

   This explains why Shader Execution Reordering (SER) yields a ~2x boost rather
   than a 16x transformation:

   * SER helps by clustering threads that access similar CSG nodes into the same
     warps. When threads share similar execution paths, their L2 cache requests hit
     the same cache lines simultaneously rather than scattering randomly.

   * However, SER cannot rewrite the fundamental instruction sequence of a deep
     recursive or iterative CSG tree traversal. If the node layout itself forces
     non-contiguous memory hops, L2 latency will remain a speed bump.

Potential Ways to Optimize CSG Layout

   * Implicit Node Flattening: Ensure your CSGNode buffers are packed in a
     strict breadth-first layout (like a heap array) so that child nodes reside in
     contiguous memory relative to their parents, maximizing L2 cache-line
     prefetching.

   * Warp-Synchronous Traversal: Grouping rays by bounding box or primitive
     type before entering the heavy CSG evaluation stage minimizes divergent node
     reads.

Want to inspect the L2 cache hit rates and memory stall counters specifically
inside your custom CSG intersection program?




L2 cache proximity to aim at
------------------------------


Small geometry size is an inherent advantage with analytic CSG::

    [lo] A[blyth@localhost CSGFoundry]$ pwd
    /home/blyth/junosw/InstallArea/Debug_g411_main/.opticks/GEOM/J26_4_1_opticks_Debug_g411/CSGFoundry
    [lo] A[blyth@localhost CSGFoundry]$ du -h *.npy
    3.1M    inst.npy
    600K    itra.npy
    1.8M    node.npy
    232K    prim.npy
    4.0K    solid.npy
    600K    tran.npy

But it comes with a major intersection complexity.
How many bytes of proximity within the L2 cache would make a difference ?


The fact that your entire geometry (node.npy at 1.8MB, tran.npy at 600K, and
inst.npy at 3.1MB) totals only a few megabytes highlights a fascinating
hardware reality: your entire JUNO analytic geometry asset fits comfortably
inside less than 8% of the RTX 5000 Ada’s 64 MB L2 cache.

Because cache capacity is not your bottleneck, the question of proximity comes
down to microarchitectural cache line granularity and warp-level working sets.



1. The Magic Number: 128 Bytes (A Single L2 Cache Line)

   * NVIDIA GPUs (Ampere, Ada Lovelace, Blackwell) fetch and manage memory in
     128-byte cache lines.

   * The Node Proximity Threshold: For tree traversal to execute with zero
     latency stalls, a parent CSGNode and its immediate left/right child nodes
     (along with their transformation matrices) must reside within the exact same
     128-byte cache line (or adjacent lines fetched in a single memory transaction).

   * If evaluating a child node requires jumping to an address outside that
     128-byte boundary, the GPU triggers an L2 cache line request (or miss),
     incurring a multi-cycle latency penalty. In a deep recursive or iterative CSG
     tree traversal, doing this repeatedly turns the execution pipeline into a
     pointer-chasing bottleneck.


::

    In [2]: 4*4*4*2
    Out[2]: 128



2. Warp-Level Proximity (1 KB to 4 KB)

   While individual node lookups are governed by the 128-byte line, a full warp
   consists of 32 threads.

   * If the 32 threads in a warp are evaluating different parts of the JUNO
     geometry (e.g., thread 0 is testing a PMT photocathode while thread 31 is
     testing the outer water tank boundary), their L2 requests scatter across
     different regions of node.npy.

   * To saturate L2 throughput, the working set for all threads in a warp during a
     single traversal step needs local proximity within a 1 KB to 4 KB subtree
     region. When threads stay within a tight memory footprint, L2 request queues
     merge, and requests are serviced concurrently without throttling.

::

    TODO : TRY SER HINT WITH THE CSGNode INDEX ? 



Why Analytic CSG Suffers Despite Small Size

Even though 5 MB is tiny, if node.npy is laid out in standard recursive pointer
order (or arbitrary creation order), a binary tree traversal results in memory
addresses bouncing back and forth across the buffer.

To make cache proximity work for analytic CSG:

1. Breadth-First / Heap Layout: Pack node.npy strictly in a breadth-first (heap
   array) order. In a heap layout, the children of node i are located at 2i and 
   2i+1, ensuring parent and child nodes land on adjacent cache lines.

2. Spatial Ordering: Reordering nodes using Morton codes (Z-order curves) based
   on their bounding box centroids ensures that geometric proximity maps directly
   to memory address proximity.



Complete Binary Tree Drawbacks
----------------------------------


The CSGNode comes from sn.h which arranges a complete binary tree layout of nodes:: 

    4815 /**
    4816 sn::GetLVNodesComplete_r
    4817 -------------------------
    4818 
    4819 Serializes tree nodes into complete binary tree vector using
    4820 0-based binary tree level order layout of nodes in the vector.
    4821 
    4822 **/
    4823 
    4824 inline void sn::GetLVNodesComplete_r(std::vector<const sn*>& nds, const sn* nd, int idx)  // static
    4825 {
    4826     assert( idx < int(nds.size()) );
    4827     nds[idx] = nd ;
    4828 
    4829     int nc = nd->num_child() ;
    4830 
    4831     if( nc > 0 && nd->is_listnode() == false ) // non-list operator node
    4832     {
    4833         assert( nc == 2 ) ;
    4834         for(int i=0 ; i < nc ; i++)
    4835         {
    4836             const sn* child = nd->get_child(i) ;
    4837 
    4838             int cidx = 2*idx + 1 + i ; // 0-based complete binary tree level order indexing
    4839 
    4840             GetLVNodesComplete_r(nds, child, cidx );
    4841         }
    4842     }
    4843 }
    4844 

That works well for simple shapes where the tree is not very deep. But for
complex shapes with deep trees lots of time is spent traversing empty ZERO
nodes. 


That code snippet highlights the classic architectural tradeoff of complete
binary tree serialization: implicit indexing vs. exponential sparsity.

Using level-order indexing (cidx = 2*idx + 1 + i) is clever because it avoids
storing explicit child pointers in the CSGNode struct—the child locations are
entirely math-derived. However, it comes with a severe structural penalty:

1. Exponential Space Bloat: A complete binary tree must allocate space for
   every node at depth h, even if branches are empty or asymmetric. If one side of
   a complex CSG tree goes 10 levels deep while the other side terminates at level
   2, thousands of dummy ZERO nodes are inserted to fill the missing slots.

2. Wasted L2 Cache Bandwidth: When the GPU traversal kernel executes, it fetches
   cache lines containing these empty ZERO nodes. Instead of loading useful
   geometry, your L2 cache bandwidth is saturated loading padding.

3. Branch & Warp Divergence: Threads evaluating deep branches get stuck
   processing long sequences of dummy nodes while threads on shallow branches
   finish early, exacerbating warp divergence.



Avoiding padding
--------------------


Alternative Layouts to Eliminate ZERO Padding

1. Compacted Node Layout with Explicit Indices (Child Offsets)

Instead of calculating child positions via 2*idx + 1, store explicit indices or
relative offsets inside the CSGNode struct::

    struct CSGNode {
        int type;          // Operator (UNION, INTERSECTION, DIFFERENCE) or Primitive
        int left_child;    // Absolute or relative index to left child
        int right_child;   // Absolute or relative index to right child
        // ... primitive parameters ...
    };

2. Linearized Postfix Evaluation (Stack Machine / Tape Layout)

For complex boolean expressions, you can flatten the CSG tree into a linear
array representing a postfix expression (Reverse Polish Notation).

* Benefit: The intersection kernel evaluates nodes sequentially using a tiny
  register-backed evaluation stack without needing recursive tree traversal or
  branch checks for null nodes.









Below is the start of the csg_tree_intersect - although POSTORDER_NEXT is just
bit twiddling the problem is accessing the typecode for each node.  That line
comparing the typecode to CSG_ZERO shows up the worst offender in ncu.

#define POSTORDER_NEXT(currIdx, elevation )( ((currIdx) & 1) ? (currIdx) >> 1 :  ((currIdx) << (elevation)) + (1 << (elevation)) )


276 TREE_FUNC
277 void intersect_tree( bool& valid_isect, float4& isect, const CSGNode* node, const float4* plan0, const qat4* itra0, const float t_min , const float3& ray_origin, const float3& ray_direction, bool dumpxyz )
278 {
279     const int numNode=node->subNum() ;   // SO THIS SHOULD NO LONGER EVER BE 1 
280     unsigned height = TREE_HEIGHT(numNode) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
281     float propagate_epsilon = 0.0001f ;  // ? 
282     int ierr = 0 ;
283 
284     LUT lut ;
285     Tranche tr ;
286     tr.curr = -1 ;
287 
288     unsigned fullTree = PACK4(0,0, 1 << height, 0 ) ;  // leftmost: 1<<height,  root:1>>1 = 0 ("parent" of root)  
302     tranche_push( tr, fullTree, t_min );
303 
304     CSG_Stack csg ;
305     csg.curr = -1 ;
309 
310     while (tr.curr > -1)
311     {
315         unsigned slice ;
316         float tmin ;    // NB different from t_min when looping 
317 
318         ierr = tranche_pop(tr, slice, tmin );
319         if(ierr) break ;
320 
321         // nodeIdx, beginIdx, endIdx are 1-based level order tree indices, root:1 leftmost:1<<height 
322         unsigned beginIdx = UNPACK4_2(slice);
323         unsigned endIdx   = UNPACK4_3(slice);
324 
331         unsigned nodeIdx = beginIdx ;
332         while( nodeIdx != endIdx )
333         {
334             unsigned depth = TREE_DEPTH(nodeIdx) ;
335             unsigned elevation = height - depth ;
336 
337             const CSGNode* nd = node + nodeIdx - 1 ;   // nodeIdx is 1-based level order index 
338             OpticksCSG_t typecode = (OpticksCSG_t)nd->typecode() ;
345 
346             if( typecode == CSG_ZERO )
347             {
348                 nodeIdx = POSTORDER_NEXT( nodeIdx, elevation ) ;
349                 continue ;
350             }
351             bool node_or_leaf = typecode >= CSG_NODE ;
355 
356             if(node_or_leaf)
357             {
358                 float4 nd_isect = make_float4(0.f, 0.f, 0.f, 0.f) ;
359 
360                 bool nd_valid_isect(false);
361                 intersect_node( nd_valid_isect, nd_isect, nd, node, plan0, itra0, tmin, ray_origin, ray_direction, dumpxyz );



Linearized Postfix Evaluation
-------------------------------

Elaborate on : Linearized Postfix Evaluation.   As recursion is not supported
within intersection but the problem of CSG tree intersection is naturally
recursive in nature the algorithm is based around post order sequences.

That line comparing typecode to CSG_ZERO is the exact symptom of the complete
binary tree's structural tax.

When your post-order traversal steps through nodeIdx using POSTORDER_NEXT,
asymmetric branches in deep JUNO subtrees land on indices that contain no
geometry. The GPU must still issue a global/L2 cache fetch for that CSGNode,
read it into registers, execute the typecode == CSG_ZERO branch, discard it,
and continue. You are burning L2 bandwidth and instruction slots on invisible
padding.



What is Linearized Postfix Evaluation (RPN Tape)?

Because recursion is forbidden inside OptiX intersection programs, the
traditional workaround is evaluating the tree using explicit post-order
sequences. However, instead of storing the tree as a sparse complete binary
array (where position is dictated by level-order math like 2i+1), Linearized
Postfix Evaluation stores the CSG tree as a dense, compact tape using Reverse
Polish Notation (RPN).

1. How the Tape is Structured

Instead of level-order padding, nodes are serialized strictly in evaluation
order (post-order). For example, a boolean expression like (Sphere A UNION
Sphere B) INTERSECTION Cylinder C becomes a flat array of nodes:


    Primitive A
    Primitive B
    OP_UNION
    Primitive C
    OP_INTERSECTION

There are zero CSG_ZERO padding nodes. Every single entry in the array is an
active primitive or a valid boolean operator.


2. How the Evaluation Loop Works in CUDA

To evaluate this tape inside your intersection kernel without recursion, each
thread maintains a tiny thread-local register stack (e.g., an array of float4
intervals of depth 4 to 8, since CSG tree depths in JUNO rarely exceed small
limits).

The execution loop becomes a simple, dense linear scan or a compact jump table::


    // Conceptual Postfix Tape Evaluation in CUDA
    float4 stack[8];
    int stack_ptr = -1;

    for (int i = 0; i < tape_length; ++i) {
        const CSGNode& node = tape[i];
        OpticksCSG_t typecode = (OpticksCSG_t)node.typecode();

        if (typecode < CSG_NODE) { 
            // 1. It's a Primitive (Sphere, Cylinder, etc.)
            float4 isect;
            bool valid = intersect_primitive(node, ray_origin, ray_direction, isect);
            stack[++stack_ptr] = isect; // Push interval onto register stack
        } 
        else {
            // 2. It's an Operator (UNION, INTERSECTION, DIFFERENCE)
            float4 right_isect = stack[stack_ptr--]; // Pop right
            float4 left_isect  = stack[stack_ptr--]; // Pop left
            
            float4 combined = combine_intervals(typecode, left_isect, right_isect);
            stack[++stack_ptr] = combined; // Push result back
        }
    }
    float4 final_result = stack[0];


Why This Eliminates Your Bottleneck

1. Zero Padding / Zero CSG_ZERO Checks: Because the tape contains only real nodes,
   every memory fetch retrieves actual geometry data or valid operators. The
   typecode == CSG_ZERO branch vanishes entirely.

2. Linear Memory Prefetching: Instead of jumping across sparse indices via
   POSTORDER_NEXT (which causes L2 cache thrashing and erratic address jumps), the
   GPU hardware prefetcher can stream contiguous cache lines linearly as the
   thread steps through the tape.

3. Register-Native Evaluation: Operating on a local register stack avoids
   complex tranche tracking and stack structures, drastically reducing register
   pressure and instruction count per intersection test.





How can we refactor the CPU-side CSG tree serialization code to generate a
dense postfix evaluation tape instead of a complete binary tree?::


    /**
    sn::GetLVNodesPostfix_r
    -------------------------

    Serializes tree nodes into a dense postfix (RPN) evaluation vector 
    using a post-order traversal (Left -> Right -> Operator).
    **/

    inline void sn::GetLVNodesPostfix_r(std::vector<const sn*>& nds, const sn* nd)
    {
        if (!nd) return;

        int nc = nd->num_child();

        // If it's a binary operator node with children, visit them first (Post-order)
        if (nc > 0 && nd->is_listnode() == false)
        {
            assert(nc == 2); // Binary CSG operator
            for (int i = 0; i < nc; ++i)
            {
                GetLVNodesPostfix_r(nds, nd->get_child(i));
            }
        }

        // Finally, append the current node (either primitive or operator)
        nds.push_back(nd);
    }




Key Structural Differences

1. Zero Padding: Unlike complete binary tree serialization which allocates
   slots for missing nodes, nds.push_back(nd) only records actual, valid nodes.
   The vector size shrinks from megabytes of sparse padding to a tight array
   containing only active primitives and boolean operators.

2. Buffer Layout: The resulting node.npy (or equivalent buffer uploaded to the
   GPU) is now a pure sequential execution tape.

How This Simplifies the GPU Kernel

With a dense postfix tape, your kernel no longer needs complex tranche stacks
or POSTORDER_NEXT bit-twiddling. Instead of jumping around indices, a thread
simply evaluates the tape from index 0 to tape_length - 1 using a thread-local
register stack:

* Primitives are intersected and pushed onto the register stack.

* Operators pop the top two intervals from the stack, combine them via boolean logic, and push the result back.




Listnodes were a workaround to reduce tree depth of some complex solids by
having a special type of compound "leaf" node within the binary tree that via
an index and count refers to a list of primatives leaf nodes stored after the
complete binary tree. Do you think the greated efficiency of RPN layout would
remove the need for such a workaround ? A very strong use of this is complex
solids with many non-overlapping holes cut out of them - the listnode very
efficiently handles the union of all those cutaways without wasting nodes on
representing a list inside a tree.  So how could RPN support listnode ? I
suppose just directly ?
  


You hit on the exact reason why listnodes were invented in the first place. In
a strict binary tree, representing a complex solid with 20 non-overlapping
holes (subtractions) requires an unbalanced chain of 19 binary UNION operators.
That balloons tree depth, consumes massive amounts of memory with intermediate
operator nodes, and causes severe warp divergence.

Listnodes were a brilliant workaround because they flattened that degenerate
binary tree into a single compound node pointing to a flat array of primitives.


1. Does RPN Remove the Need for Listnodes?

RPN doesn't just remove the workaround—it naturally subsumes and supersedes it.

The only reason listnodes were necessary in Opticks is that complete binary
trees enforce a strict dyadic (left/right child) constraint. RPN, by its very
nature as a linear evaluation tape, is not restricted to binary operations. It
natively supports N-ary operations.

2. How RPN Supports Listnodes Directly (and Cleaner)

You guessed it: directly, but without needing a separate side-array stored
"after the tree."

In an RPN tape layout, a listnode (like a multi-hole cutaway) becomes a
first-class citizen using an N-ary Reduction Operator. Instead of jumping to a
disjoint memory region to fetch secondary primitives, the tape can execute a
compact sequence:

* Evaluate Primitives sequentially: Evaluate Primitive 1, Primitive 2, ...,
  Primitive N, pushing their resulting intervals onto the thread-local register
  stack.

* Execute an N-ary Operator: Hit a specialized operator instruction on the
  tape—for example, OP_N_UNION with a payload of count = N.

* Stack Reduction: The evaluation loop pops the top N intervals from the
  register stack, merges them into a single combined hole interval in one pass,
  and pushes 1 result back.


Conceptual RPN Tape for a Multi-Hole Solid::

    [ Base Solid Primitive ]
    [ Hole Primitive 1 ]
    [ Hole Primitive 2 ]
    [ Hole Primitive 3 ]
    [ OP_N_UNION (count = 3) ]  <-- Pops 3 holes, unions them into one compound cutter
    [ OP_DIFFERENCE ]           <-- Subtracts the combined holes from the base solid


Why This is Superior for L2 Cache Performance

* Zero Pointer Chasing: In the old design, a listnode required reading a node
  inside the binary tree, extracting an index/count, and then jumping to a
  completely different buffer address to read the list of primitive leaf nodes.
  That destroys L2 cache locality.

* Contiguous Streaming: In an RPN tape, the primitives belonging to the listnode
  and the reduction operator are stored contiguously inline on the linear
  execution tape. The GPU prefetcher streams them sequentially into the L2 cache
  with zero random access overhead.









