asyncio
==========

Async functionalities

1. DONE: receive udp messages to config GUI rendering via BCfg::liveline
2. Viz server that receives NPY gensteps and presents simulated propagations 
3. Compute server that recieve NPY gensteps and responds with NPY hits 


"np.h" : header only deployment of array+metadata send/receive functionality
------------------------------------------------------------------------------

* depending only on "boost/asio.hpp" (which is header only)
* so any job gets access to Opticks processing just by including a single header
  that enables collected gensteps to be sent to the server and hits returned


Hmm what about nlohmann::json which is the basis of npy/NMeta.

* avoid depending on this, metadata should just be a std::string that 
  happens to be json, defaulting to "{}" 


Where to put this implementation ?
-------------------------------------

Experimenting in ~/opticks/examples/UseBoostAsioNPY

* hmm depending on Opticks NPY is probably not appropriate 
* do not need all the NPY bells and whistles and aiming 
  for a simple header only implementation so aim lower level 
* this fits with the very general level of this functionality, 
  it is way beneath Opticks

Better to target http://github.com:simoncblyth/np.git  ~/np/

See: ~/np/tests/NPNetTest.cc



DONE : UDP liveline config : sends "commandline" options over UDP 
--------------------------------------------------------------

New implementation based on boost asio with::

   brap/BListenUDP
   brap/BCfg
   okg/OpticksHub 
   oglrap/OpticksViz
   env/bin/udp.py 

See also the early stages::
  
    examples/UseBoostAsioUDP


NPY transport : how to do it 
-------------------------------

The primordial approach used asio + ZMQ + asio-zmq project to do this. But that 
project appears dead and doesnt compile with current boost.


Manual TCP framing ? Instead of using ZMQ : DONE with simple 16 byte prefix holding hdr/arr/meta/spare sizes
-------------------------------------------------------------------------------------------------------------------

* :google:`tcp message framing`

* https://blog.stephencleary.com/2009/04/message-framing.html

  * "4-byte unsigned little-endian" length header prefix seems sensible and easier than using delimiters 
     which entails escaping and unescaping


Boost Beast : websocket and http on top of boost asio
---------------------------------------------------------

* https://www.boost.org/doc/libs/1_74_0/libs/beast/doc/html/index.html


NPY transport using python socket and TCP 
---------------------------------------------

* ~/opticks/bin/npy.py


NPY transport with ZMQ socket.send_multipart socket.recv_multipart
--------------------------------------------------------------------

* https://pyzmq.readthedocs.io/en/latest/api/zmq.html


env/zeromq/pyzmq/npycontext.py subclasses zmq.Socket::

    007 import zmq
     ...
     21 class NPYSocket(zmq.Socket):
     31     def send_npy(self, a, flags=0, copy=False, track=False, ipython=False):
     ...
     52             stream = io.BytesIO()
     53             np.save( stream, a )     # write ndarray into stream
     54             stream.seek(0)
     55             content = stream.read()  # NPY format serialized bytes 
     56             buf = memoryview(content)
     57             bufs.append(buf)
     58             if hasattr(a, 'meta'):
     59                 for jsd in a.meta:
     60                     bufs.append(json.dumps(jsd))   # convert dicts to json strings
     61                 pass
     62             pass
     ..
     71         log.info("send_npy sending %s bufs copy %s " % (len(bufs),copy))
     72         return self.send_multipart( bufs, flags=flags, copy=copy, track=track)
           
    075     def recv_npy(self, flags=0, copy=False, track=False, meta_encoding="ascii", ipython=False):
    ...
    105         if copy:
    106             log.warn("using slower copy=True option ")
    107             assert 0
    108             msgs = self.recv_multipart(flags=flags, copy=True, track=track)  # bytes
    109             bufs = map(lambda msg:buffer(msg),msgs)
    110         else:
    111             frames = self.recv_multipart(flags=flags,copy=False, track=track)
    112             bufs = map(lambda frame:frame.buffer, frames)            # memoryview object 
    113         pass
    114 
    115         if ipython:
    116             log.info("stopped in recv_npy just after receiving the bufs (list of memoryview)")
    117             IPython.embed()
    118         pass
    119 
    120         arys = []
    121         meta = []
    122         other = []
    123 
    124         for buf in bufs:
    125             stream = io.BytesIO(buf)     # file like access to memory buffer
    126             peek = stream.read(1)
    127             stream.seek(0)
    128             if peek == '\x93':
    129                 a = np.load(stream)
    130                 aa = a.view(NP)          # view as subclass, to enable attaching metadata
    131                 arys.append(aa)
    132             else:
    133                 txt = codecs.decode(stream.read(-1))
    134                 if peek == '{':
    135                     try:
    136                         jsdict = json.loads(txt)
    137                     except ValueError:
    138                         log.warn("JSON load error for %s " % repr(txt))
    139                     pass
    140                     meta.append(jsdict)
    141                 else:
    142                     other.append(txt)
    143                 pass
    144             pass


env/zeromq/pyzmq/npysend.py uses the socket subclass to send_npy recv_npy::

    093 class NPYProcessor(object):
     94     def __init__(self, config):
     95         self.config = config
     96 
     97     def cerenkov(self, tag):
     98         return self.load(tag, "cerenkov")
     99 
    100     def scintillation(self, tag):
    101         return self.load(tag, "scintillation")
    102 
    103     def photon(self, tag):
    104         return self.load(tag, "photon")
    105 
    106     def process(self, request):
    107         context = NPYContext()
    108         socket = context.socket(zmq.REQ)
    109         log.info("connect to endpoint %s " % self.config.endpoint )
    110         socket.connect(self.config.endpoint)
    111         log.info("send_npy")
    112         socket.send_npy(request,copy=self.config.copy,ipython=self.config.ipython)
    113         response = socket.recv_npy(copy=self.config.copy, ipython=self.config.ipython)
    114         log.info("response %s\n%s " % (str(response.shape), repr(response)))
    115 
    116         meta = getattr(response, 'meta', [])
    117         for jsd in meta:
    118             print pprint.pformat(jsd)
    119         pass
    120         return response





trawling env for old concurrency and message queue experiments
----------------------------------------------------------------

env/rootmq/include/EvMQ.h
    message queue monitor 
    
env/rootmq/include/rootmq.h
    amqp.h based C interface to message queue 

env/rootmq/include/MQ.h
    131 R__EXTERN MQ* gMQ ; 

    gMQ interface for sending receiving messages with control over a monitoring thread 

env/rootmq/src/MQ.cc
     
    527 void MQ::StartMonitorThread()
    528 {
    529    if(!fConfigured) this->Configure();
    530    fMonitorRunning = kTRUE ;
    531    rootmq_basic_consume_async( fQueue.Data() );
    532    //rootmq_basic_consume( fQueue.Data() );  // ....  dont spin off thread (means that GUI doesnt update) ... BUT useful to check if threading is the cause of issues
    533 }

env/rootmq/src/rootmq_collection.c

   glib based 




OptiX 6 host thread safety
--------------------------

::

    optix7-;optix6-p 123   # optix 6.5 pdf 

Currently, the OptiX host API is not guaranteed to be thread-safe. While it may
be successful in some applications to use OptiX contexts in different host
threads, it may fail in others. OptiX should therefore only be used from within
a single host thread.

**HMM: need to make the jump to optix 7 before attempting the compute server**


OptiX 7 host thread safety
-----------------------------

::

    optix7-;optix7-p 7

3.2 Thread safety
Almost all host functions are thread-safe. Exceptions to this rule are
identified in the API documentation. A general requirement for thread-safety is
that output buffers and any temporary or state buffers are unique. For example,
you can build more than one acceleration structure concurrently from the same
input geometry, as long as the temporary and output device memory are disjoint.
Temporary and state buffers are always part of the parameter list if they are
needed to execute the method.


p49: Ray generation launches

To initiate a pipeline launch, use the optixLaunch function. All launches are
asynchronous, using CUDA streams. When it is necessary to implement
synchronization, use the mechanisms provided by CUDA streams and events.

p50::

    CUstream stream;
    cuStreamCreate(&stream);
    CUdeviceptr raygenRecord, hitgroupRecords;
    ...
    Generate acceleration structures and SBT records
    unsigned int width = ...;
    unsigned int height = ...;
    unsigned int depth = ...;
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = raygenRecord;
    sbt.hitgroupRecords = hitgroupRecords;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = numHitGroupRecords;
    MyPipelineParams pipelineParams = ...;
    CUdeviceptr d_pipelineParams;
    ...
    Allocate and copy the params to the device
    optixLaunch(pipeline, stream,
       d_pipelineParams, sizeof(MyPipelineParams),
       &sbt, width, height, depth);


strategy for compute server
------------------------------

* no point of anything more than "long range learning" work on this prior to moving to optix 7 

Things to investigate:

* CUstream

CUDA Streams
---------------


* :google:`cuda multiple host threads`

* http://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
* ~/opticks_refs/CUDAStreamsAndConcurrencyWebinar.pdf
* this looks rather old

* https://www.olcf.ornl.gov/wp-content/uploads/2020/07/07_Concurrency.pdf
* ~/opticks_refs/CUDA_Concurrency_Bob_Crovella_07_2020.pdf

cudaLaunchHostFunc
    add host callback for when a stream completes
    (Uses a thread spawned by the GPU driver to perform the work)



requirements for a compute server 
----------------------------------

0. populate GPU context with geometry at initialization, reuse that context for processing 
1. accept gensteps and return hits to caller via network (boost::asio server) 
2. queue gensteps when some are received whilst a launch is already in progress
   (could investigate concurrent launches when have multiple GPUs but the number will be small)
3. json responses to json status queries, eg returning number in queue 



