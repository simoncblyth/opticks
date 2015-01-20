CUDA Atomic
==============


* :google:`atomicMin atomicAdd atomicMax atomicExch atomicOr`


atomic funcs in chroma
------------------------

::

    simon:cuda blyth$ grep atomic *.*
    daq.cu:             atomicMin(earliest_time_int + channel_index, time_int);
    daq.cu:             atomicAdd(channel_q_int + channel_index, charge_int);
    daq.cu:             atomicOr(channel_histories + channel_index, history);
    daq.cu:     atomicMin(earliest_time_int + channel_offset, time_int);
    daq.cu:     atomicAdd(channel_q_int + channel_offset, charge_int);
    daq.cu:     atomicOr(channel_histories + channel_offset, history);
    hybrid_render.cu:       data = atomicExch(addr, data+atomicExch(addr, 0.0f));
    pdf.cu: atomicMin(&distance_table_len, i);
    propagate.cu:       atomicAdd(&counter, 1);
    propagate.cu:   atomicAdd(index_counter, counter);
    propagate.cu:   int offset = atomicAdd(index_counter, 1);
    propagate.cu:   int out_idx = atomicAdd(output_queue, 1);



Old GPU capability compatibility code
--------------------------------------

* https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks

Capability <2.0 (but >=1.3)  

::

    // For capability <2.0 (but >=1.3) GPUs
     #define ATOMIC_ADD(x,y) MyAtomicAdd (x, (float)(y))
     #else
     #define ATOMIC_ADD(x,y) atomicAdd (x, (float)(y))
     #endif
