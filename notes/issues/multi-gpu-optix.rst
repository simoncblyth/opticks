multi-gpu-optix
==================



6.0.0 Manual 
---------------

* https://raytracing-docs.nvidia.com/optix/guide/index.html#performance#13001


Use a separate buffer copy per device in a multi-GPU environment.

In multi-GPU environments, INPUT_OUTPUT buffers may be stored on the
device, with a separate copy per device by using the RT_BUFFER_GPU_LOCAL buffer
attribute. This is useful for avoiding the slower reads and writes by the
device to host memory. RT_BUFFER_GPU_LOCAL is useful for scratch buffers, such
as random number seed buffers and variance buffers.

TODO: review my buffers
~~~~~~~~~~~~~~~~~~~~~~~~

OptiX Forum 
---------------

* https://devtalk.nvidia.com/search/more/sitecommentsearch/multiple%20GPUs/?boards=254


Posted 11/08/2018 01:54 PM  Detlef Roettger  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/1043859/optix/very-poor-multi-gpu-scaling-on-dgx-1/post/5311496/#5311496


When using multiple devices in OptiX the output and input_output buffers reside
in pinned memory and there is congestion when writing over the PCI-E bus to the
same target with many GPUs.

If your renderer is accumulating images, that expensive read-modify-write
operation can be done in GPU local buffers and only the final result can be
written to an output buffer which then resides in pinned memory. That should
increase the multi-GPU scaling drastically.

Find some more information when digging through all links in this and the
referenced threads:
https://devtalk.nvidia.com/default/topic/1036340/?comment=5264830


Posted 06/08/2018 12:09 PM   Detlef Roettger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, output and input_output buffers reside in pinned memory on the host for
multi-GPU OptiX contexts and all GPUs access it directly via PCI-E, so there is
some congestion when doing that with many GPUs at once. Two scale nicely
though.

To avoid most of that congestion there is an RT_BUFFER_GPU_LOCAL flag which can
only be used with input_output buffers and these are then per GPU but cannot be
read by the host, so these are perfect for local accumulations on multi-GPU and
then finally writing the result into a pinned memory output buffer.

OptiX Programming Guide about that: Buffers:
http://raytracing-docs.nvidia.com/optix/guide/index.html#host#3140 Performance
guidelines touching on multi-GPU:
http://raytracing-docs.nvidia.com/optix/guide/index.html#performance#13001
Setting multiple devices:
http://raytracing-docs.nvidia.com/optix/guide/index.html#host#3002 Zero-copy
(pinned) memory when using multi-GPU:
http://raytracing-docs.nvidia.com/optix/guide/index.html#cuda#9024

Forum posts explaining that in more detail (including corrections to my own
statements because I thought the multi-GPU load balancer was not static):
https://devtalk.nvidia.com/default/topic/1030457/?comment=5242078 and follow
the links in there as well.
https://devtalk.nvidia.com/default/topic/1024818/?comment=5214480



post from Detlef Roettger Posted 10/09/2017 08:16 AM   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/1024818/optix/question-about-handling-buffers-when-using-multiple-gpus-/post/5213313/#5213313


Multi-GPU support with OptiX is more or less automatic.

Though if you're using "Interoperability with CUDA" as described in Chapter 7
of the OptiX Programming Guide, that requires that you know exactly what you're
doing with CUDA device pointers. That's not automatic and not part of the
answers below.

You can select which GPU devices to use with the rtContextSetDevices()
function.  **The default OptiX behavior is to use all available devices**.  In
versions OptiX 3 and lower it used the compatible devices of the highest
streaming multi-processor versions.  In OptiX 4 that limitation has been lifted
and it can be a heterogeneous setup of different GPU architectures (Kepler and
newer) as well. Though that can result in longer kernel compilation (per
different architecture).

**Input buffers are uploaded to all GPUs**.  Input_output and output buffers are
put into host memory and all GPUs access that via the PCI-E directly (see
Chapter 7 and 11), which adds a limit on the possible scaling progression. I
*would not recommend to use more than four GPUs per context*.  The workload
balancing does not take the number of PCI-E lanes into account. Best scaling
happens with symmetrical setups, e.g. all boards in 16x lanes slots.

Indexing is unchanged. (Always use operator[] to access buffer elements inside
device code. Pointer arithmetic is illegal.)

Atomics do not work across GPUs. Don't use them to output your final results on
multi-GPU contexts.

But input_output buffers can be flagged to be local on each GPU with this flag
(see OptiX API Reference): "RT_BUFFER_GPU_LOCAL - An RT_BUFFER_INPUT_OUTPUT has
separate copies on each device that are not synchronized". Good for temporary
scratch space.

OpenGL interop will not be active in a multi-GPU setup because output buffers
are on the host.

Other than that, read the OptiX Programming Guide Chapter 11 Performance
Guidelines which contains some more notes about multi-GPU behavior.

Then there is remote multi-GPU rendering via OptiX' progressive API available
on NVIDIA VCA clusters, which is a separate topic. You shouldn't care about
that unless you have access to a VCA.






