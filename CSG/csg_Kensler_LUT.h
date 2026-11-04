#pragma once
/**
csg_Kensler_LUT.h
==================

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define LUT_METHOD __device__ __forceinline__ 
#else
   #define LUT_METHOD inline
#endif 


struct Kensler_LUT 
{
    // We combine the original 4 unsigned ints into two 64-bit unsigned long long constants.
    // Low 64-bits = index 0 and 1 combined. High 64-bits = index 2 and 3 combined.
    // Example for ACloser: 
    //   Low  = (0x00014014ULL << 32) | 0x22121141ULL;
    //   High = (0x00000000ULL << 32) | 0x00141141ULL;
    
    static constexpr unsigned long long lutA_low  = 0x0001401422121141ULL;
    static constexpr unsigned long long lutA_high = 0x0000000000141141ULL;

    static constexpr unsigned long long lutB_low  = 0x0002205522115122ULL;
    static constexpr unsigned long long lutB_high = 0x0000000000133155ULL;

    template <bool ACloser>
    static LUT_METHOD int lookup(OpticksCSG_t operation, IntersectionState_t stateA, IntersectionState_t stateB)
    {   
        unsigned offset = 3 * (unsigned)stateA + (unsigned)stateB; 
        if (offset >= 8) return CTRL_RETURN_MISS;

        unsigned index = (unsigned)operation - (unsigned)CSG_UNION; 

        // 1. Determine whether we look at the lower half (index 0,1) or upper half (index 2,3)
        // Since index < 4, this single check splits the entire 128-bit table perfectly.
        unsigned long long target_64bit = ACloser ? 
            ((index < 2) ? lutA_low : lutA_high) : 
            ((index < 2) ? lutB_low : lutB_high);

        // 2. Adjust the index relative to the chosen 64-bit boundary
        // If index was 2 or 3, it becomes 0 or 1.
        unsigned local_index = index & 1; 

        // 3. Compute the exact global bit offset within the 64-bit word
        unsigned bit_pos = (local_index * 32) + (offset * 4);

        // 4. Use 64-bit Bit Field Extract intrinsic.
        // Extracts 4 bits from 'target_64bit' starting at 'bit_pos' in a single operation.

#if defined(__CUDACC__) || defined(__CUDABE__)
        return (int)__bfe64(target_64bit, bit_pos, 4);
#else
        return (int)((target_64bit >> bit_pos) & 0x0FULL);
#endif
    }
};


