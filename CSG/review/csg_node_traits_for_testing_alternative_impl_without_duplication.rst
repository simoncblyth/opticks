csg_node_traits_for_testing_alternative_impl_without_duplication.rst
=====================================================================

Aim : add intersect ontop rpn node described prim/list/tree without duplicated code
------------------------------------------------------------------------------------

1. replace CSGNode from csg_intersect_leaf_*.h  with quad4& q0, q1 that both CSGNode.h and sparm.h share
2. OR reconstruct CSGNode for the rpn impl ?



I want to generalize my NVIDA OptiX analytic CSG interesection impl to work
with an ultra-lite CSG node impl srpn.h in addition to the old CSGNode
approach.  Is templating the method on the node type the way to go ? And using
template specializations to handle the deviations on where parameters come from
for example.

::

    LEAF_FUNC
    void intersect_leaf(bool& valid_isect, float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction, bool dumpxyz )
    {
        valid_isect = false ;
        isect.x = 0.f ;
        isect.y = 0.f ;
        isect.z = 0.f ;
        isect.w = 0.f ;

        const unsigned typecode = node->typecode() ;
        const unsigned gtransformIdx = node->gtransformIdx() ;
        const bool complement = node->is_complement();

        const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None

        float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin ;
        float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction ;

    #if !defined(PRODUCTION) && defined(DEBUG_RECORD)
        printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx );
    #endif

    #if !defined(PRODUCTION) && defined(DEBUG)
        //printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx );
        //printf("//intersect_leaf ray_origin (%10.4f,%10.4f,%10.4f) \n",  ray_origin.x, ray_origin.y, ray_origin.z );
        //printf("//intersect_leaf ray_direction (%10.4f,%10.4f,%10.4f) \n",  ray_direction.x, ray_direction.y, ray_direction.z );
        /*
        if(q)
        {
            printf("//intersect_leaf q.q0.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q0.f.x,q->q0.f.y,q->q0.f.z,q->q0.f.w  );
            printf("//intersect_leaf q.q1.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q1.f.x,q->q1.f.y,q->q1.f.z,q->q1.f.w  );
            printf("//intersect_leaf q.q2.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q2.f.x,q->q2.f.y,q->q2.f.z,q->q2.f.w  );
            printf("//intersect_leaf q.q3.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q3.f.x,q->q3.f.y,q->q3.f.z,q->q3.f.w  );
            printf("//intersect_leaf origin (%10.4f,%10.4f,%10.4f) \n",  origin.x, origin.y, origin.z );
            printf("//intersect_leaf direction (%10.4f,%10.4f,%10.4f) \n",  direction.x, direction.y, direction.z );
        }
        */
    #endif

        switch(typecode)
        {
            case CSG_SPHERE:           intersect_leaf_sphere(           valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
            case CSG_ZSPHERE:          intersect_leaf_zsphere(          valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
            case CSG_CYLINDER:         intersect_leaf_cylinder(         valid_isect, isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
            case CSG_BOX3:             intersect_leaf_box3(             valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
            case CSG_CONE:             intersect_leaf_newcone(          valid_isect, isect, node->q0,               t_min, origin, direction ) ; break ;
            case CSG_CONVEXPOLYHEDRON: intersect_leaf_convexpolyhedron( valid_isect, isect, node, plan,             t_min, origin, direction ) ; break ;





With template specialization of full funcs like below you end up duplicating
dozens of lines of transform logic (q->right_multiply) and debug blocks just
because SRPNNode stores its typecode or gtransformIdx in a bitfield instead of
a member function::


    template <typename NodeType>
    void intersect_leaf(...) { /* ... */ }

    template <>
    void intersect_leaf<SRPNNode>(...) { /* duplicate all switch/transform code here? */ }




Recommended Pattern: Accessor Helpers & Traits

Instead of specializing the heavy intersection function, decouple the data
extraction using inline helper functions or a traits struct.

Here is how you can refactor it cleanly:


::

    // For traditional CSGNode
    __device__ __forceinline__ unsigned getNodeTypecode(const CSGNode* node) {
        return node->typecode();
    }
    __device__ __forceinline__ unsigned getNodeGTransformIdx(const CSGNode* node) {
        return node->gtransformIdx();
    }
    __device__ __forceinline__ bool getNodeComplement(const CSGNode* node) {
        return node->is_complement();
    }
    __device__ __forceinline__ void getNodeParams(const CSGNode* node, float4& q0, float4& q1) {
        q0 = node->q0;
        q1 = node->q1;
    }

    // For ultra-lite srpn.h nodes (adjust based on srpn struct layout)
    __device__ __forceinline__ unsigned getNodeTypecode(const srpn_node* node) {
        return node->get_typecode(); // or bit-shift extraction
    }
    __device__ __forceinline__ unsigned getNodeGTransformIdx(const srpn_node* node) {
        return node->get_transform_index();
    }
    __device__ __forceinline__ bool getNodeComplement(const srpn_node* node) {
        return node->is_complement();
    }
    __device__ __forceinline__ void getNodeParams(const srpn_node* node, float4& q0, float4& q1) {
        // srpn nodes might fetch parameters from a separate buffer if ultra-lite
        q0 = unpack_q0(node);
        q1 = unpack_q1(node);
    }


::

    template <typename NodeType>
    LEAF_FUNC void intersect_leaf(bool& valid_isect, float4& isect, const NodeType* node, const float4* plan, const qat4* itra, const float t_min, const float3& ray_origin, const float3& ray_direction, bool dumpxyz)
    {
        valid_isect = false;
        isect = make_float4(0.f, 0.f, 0.f, 0.f);

        const unsigned typecode = getNodeTypecode(node);
        const unsigned gtransformIdx = getNodeGTransformIdx(node);
        const bool complement = getNodeComplement(node);

        const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr;
        float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin;
        float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction;

        float4 q0, q1;
        getNodeParams(node, q0, q1); // Fetches fields dynamically based on node type

        switch(typecode)
        {
            case CSG_SPHERE:           intersect_leaf_sphere(           valid_isect, isect, q0,       t_min, origin, direction ); break;
            case CSG_ZSPHERE:          intersect_leaf_zsphere(          valid_isect, isect, q0, q1,   t_min, origin, direction ); break;
            case CSG_CYLINDER:         intersect_leaf_cylinder(         valid_isect, isect, q0, q1,   t_min, origin, direction ); break;
            // ...
        }
    }



Key OptiX / CUDA Considerations

1. Always use __forceinline__: Ensure every accessor and leaf intersection
   primitive is marked __forceinline__ so the NVCC compiler strips away function
   call overhead.

2. Watch Register Pressure: Ultra-lite nodes (srpn.h) are usually designed to save
   GPU memory bandwidth by packing fields into fewer bytes. Ensure your unpacking
   functions don't introduce unnecessary intermediate registers inside the
   ray-tracing loop.



Step 1: Define the Traits Structs

Create separate policy structs that wrap the specific layout details, bitfield
extractions, or parameter lookups for each node type.



    // 1. Traits for traditional CSGNode
    struct CSGNodeTraits {
        using NodePtr = const CSGNode*;

        __device__ __forceinline__ static unsigned getTypecode(NodePtr node) {
            return node->typecode();
        }

        __device__ __forceinline__ static unsigned getGTransformIdx(NodePtr node) {
            return node->gtransformIdx();
        }

        __device__ __forceinline__ static bool getComplement(NodePtr node) {
            return node->is_complement();
        }

        __device__ __forceinline__ static void getParams(NodePtr node, float4& q0, float4& q1) {
            q0 = node->q0;
            q1 = node->q1;
        }
    };

    // 2. Traits for ultra-lite srpn.h nodes
    // (Adjust member names to match whatever packed layout srpn.h uses)
    struct SRPNNodeTraits {
        using NodePtr = const srpn_node*;

        __device__ __forceinline__ static unsigned getTypecode(NodePtr node) {
            // Example: extracted via bit-shifting or packed field
            return node->typecode;
        }

        __device__ __forceinline__ static unsigned getGTransformIdx(NodePtr node) {
            return node->transform_idx;
        }

        __device__ __forceinline__ static bool getComplement(NodePtr node) {
            return (node->flags & SRPN_COMPLEMENT_MASK) != 0;
        }

        __device__ __forceinline__ static void getParams(NodePtr node, float4& q0, float4& q1) {
            // If srpn nodes store indices or packed data instead of full float4 fields directly:
            q0 = make_float4(node->p[0], node->p[1], node->p[2], node->p[3]);
            q1 = make_float4(node->p[4], node->p[5], node->p[6], node->p[7]);
        }
    };





Step 2: Write the Templated Intersection Function

By parameterizing intersect_leaf on the Traits class, the function body remains
agnostic of whether it is dealing with a full CSGNode or an ultra-lite
srpn_node.::


    template <typename Traits>
    LEAF_FUNC void intersect_leaf(
        bool& valid_isect,
        float4& isect,
        typename Traits::NodePtr node,
        const float4* plan,
        const qat4* itra,
        const float t_min,
        const float3& ray_origin,
        const float3& ray_direction,
        bool dumpxyz
    )
    {
        valid_isect = false;
        isect = make_float4(0.f, 0.f, 0.f, 0.f);

        // Query properties through the traits policy class
        const unsigned typecode = Traits::getTypecode(node);
        const unsigned gtransformIdx = Traits::getGTransformIdx(node);
        const bool complement = Traits::getComplement(node);

        const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr;
        float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin;
        float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction;

        float4 q0, q1;
        Traits::getParams(node, q0, q1);

    #if !defined(PRODUCTION) && defined(DEBUG_RECORD)
        printf("//[intersect_leaf typecode %d gtransformIdx %d \n", typecode, gtransformIdx);
    #endif

        switch(typecode)
        {
            case CSG_SPHERE:
                intersect_leaf_sphere(valid_isect, isect, q0, t_min, origin, direction);
                break;
            case CSG_ZSPHERE:
                intersect_leaf_zsphere(valid_isect, isect, q0, q1, t_min, origin, direction);
                break;
            case CSG_CYLINDER:
                intersect_leaf_cylinder(valid_isect, isect, q0, q1, t_min, origin, direction);
                break;
            case CSG_BOX3:
                intersect_leaf_box3(valid_isect, isect, q0, t_min, origin, direction);
                break;
            case CSG_CONE:
                intersect_leaf_newcone(valid_isect, isect, q0, t_min, origin, direction);
                break;
            case CSG_CONVEXPOLYHEDRON:
                // If convex polyhedron needs the raw pointer instead of unpacked float4s:
                intersect_leaf_convexpolyhedron(valid_isect, isect, node, plan, t_min, origin, direction);
                break;
        }
    }


Step 3: Invoking the Templates

When dispatching from your OptiX closest-hit or intersection program, you
simply instantiate the template with the appropriate traits::

    // For an old-style CSGNode tree traversal
    intersect_leaf<CSGNodeTraits>(valid, isect, csg_node_ptr, plan, itra, t_min, ray_o, ray_d, dumpxyz);

    // For an ultra-lite srpn.h tree traversal
    intersect_leaf<SRPNNodeTraits>(valid, isect, srpn_node_ptr, plan, itra, t_min, ray_o, ray_d, dumpxyz);


















