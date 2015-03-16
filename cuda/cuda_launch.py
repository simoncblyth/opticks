#!/usr/bin/env python
"""
cuda_launch.py
================

Simple Tool to plan CUDA launch sequence
extending the CUDA model upwards to become::

     launch 
       grid
         block 
           thread


Iterators provide the appropriate offsets to 
use when using multiple launches to deal with 
single large 1D or 2D problems. 


TODO
----

#. make 1D usage more like 2D


Usage examples
---------------

NB separate options are used to configure the  1D and 2D launches::

    (chroma_env)delta:cuda blyth$ cuda_launch.py -d2 1024 768 --launch 2,2,1 --block 16,16,1
    Launch2D work [1024, 768] [786432]  launch  [2, 2, 1] [4]  block [16, 16, 1] [256] 
    launch_index 0 work_count 196608 offset (0, 0)               grid    (32, 24, 1) [768]     block    (16, 16, 1) [256] 
    launch_index 1 work_count 196608 offset (512, 0)             grid    (32, 24, 1) [768]     block    (16, 16, 1) [256] 
    launch_index 2 work_count 196608 offset (0, 384)             grid    (32, 24, 1) [768]     block    (16, 16, 1) [256] 
    launch_index 3 work_count 196608 offset (512, 384)           grid    (32, 24, 1) [768]     block    (16, 16, 1) [256] 
    (chroma_env)delta:cuda blyth$ 

    (chroma_env)delta:cuda blyth$ cuda_launch.py -d1 1024 768 --threads-per-block 256 --max-blocks 768  
    Launch1D work [1024, 768] total 786432 max_blocks 768 threads_per_block 256 block (256, 1, 1) 
    offset          0 count 196608 grid (768, 1) block (256, 1, 1) 
    offset     196608 count 196608 grid (768, 1) block (256, 1, 1) 
    offset     393216 count 196608 grid (768, 1) block (256, 1, 1) 
    offset     589824 count 196608 grid (768, 1) block (256, 1, 1) 
    (chroma_env)delta:cuda blyth$ 


CUDA Launch config basics
--------------------------

* http://cs.nyu.edu/courses/spring12/CSCI-GA.3033-012/lecture5.pdf
* http://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
* http://www.martinpeniak.com/index.php?option=com_content&view=article&catid=17:updates&id=288:cuda-thread-indexing-explained
* :google:`mapping cuda thread index to pixel`
* http://stackoverflow.com/questions/9099749/using-cuda-to-find-the-pixel-wise-average-value-of-a-bunch-of-images
* https://devtalk.nvidia.com/default/topic/400713/simple-question-about-blockidx-griddim/
 

Launch constraints
~~~~~~~~~~~~~~~~~~~
 
#. block of threads: max dimensions (1024, 1024, 64), BUT max threads per block 1024 eg (32,32,1)
#. launch runtime limit is the kicker, as a result split into multiple launches controlled by launch_dim

#. grid of blocks, max block dimensions (2147483647, 65535, 65535) : hit runtime limit long before hitting these 
#. also use multiples of 32 for block of threads dimensions to work better with the hardware

deviceQuery::

        Device 0: "GeForce GT 750M"
          CUDA Driver Version / Runtime Version          5.5 / 5.5
          CUDA Capability Major/Minor version number:    3.0
          Total amount of global memory:                 2048 MBytes (2147024896 bytes)
          ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores
          ...
          Total amount of constant memory:               65536 bytes
          Total amount of shared memory per block:       49152 bytes
          Total number of registers available per block: 65536
          Warp size:                                     32
          Maximum number of threads per multiprocessor:  2048
          Maximum number of threads per block:           1024
          Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
          Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
          Maximum memory pitch:                          2147483647 bytes
          ...
          Run time limit on kernels:                     Yes
          ...


practical limits
~~~~~~~~~~~~~~~~

On OSX, need to restrict kernel launch times to less than 5 seconds, otherwise they get 
killed, GPU panics and hard system crashes result. Upshot is must restrict the 
number of blocks within the grid and split expensive processing into multiple launches 
to keep each launch within the timeout.
 

"""
import os, logging, argparse
log = logging.getLogger(__name__)

from env.cuda.cuda_profile_parse import Parser

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

from operator import mul
import numpy as np

mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # integer division trick, rounding up without iffing around
rep_ = lambda _:"%10s [%s] " % (str(_),mul_(_))
ivec_ = lambda _:map(int,_.split(","))


def chunk_iterator(nelements, nthreads_per_block=64, max_blocks=1024):
    """
    Extracted from chroma.gpu.tools  

    Iterator that yields tuples with the values requried to process
    a long array in multiple kernel passes on the GPU.

    Each yielded value is of the form,
        (first_index, elements_this_iteration, nblocks_this_iteration)

    Example:
        >>> list(chunk_iterator(300, 32, 2))
        [(0, 64, 2), (64, 64, 2), (128, 64, 2), (192, 64, 2), (256, 9, 1)]
    """
    first = 0 
    while first < nelements:
        elements_left = nelements - first
        blocks = int(elements_left // nthreads_per_block)
        if elements_left % nthreads_per_block != 0:
            blocks += 1 # Round up only if needed
        blocks = min(max_blocks, blocks)
        elements_this_round = min(elements_left, blocks * nthreads_per_block)

        yield (first, elements_this_round, blocks)
        first += elements_this_round



def launch_iterator_2d(work_dim_=(1024,768,1), launch_dim_=(2,2,1), block_dim_=(8,8,1)):
    """

    #. numpy array division is elementwise, and integer division is used when inputs are dtype int
       (unlike with python the integer division "//" are not necessary, merely indicative ) 

    """
    work_dim, launch_dim, block_dim = map(lambda _:np.array(_,dtype=int), (work_dim_, launch_dim_, block_dim_))
    if len(work_dim) == 2:
        work_dim = np.append(work_dim, 1)

    work_per_launch = reduce(div_, (work_dim,launch_dim) )

    grid_dim = reduce(div_, (work_dim,launch_dim,block_dim) )   # work_dim//launch_dim//block_dim  taking care of roundup


    launch_index = 0
    for launchIdx_y in range(launch_dim[1]):
        launch_offset_y = work_per_launch[1]*launchIdx_y
        for launchIdx_x in range(launch_dim[0]):
            launch_offset_x = work_per_launch[0]*launchIdx_x
            launch_offset = (launch_offset_x, launch_offset_y )
            launch_work = mul_(grid_dim)*mul_(block_dim)
            yield launch_index, launch_work, launch_offset, tuple(grid_dim.tolist()), tuple(block_dim.tolist()) 
            launch_index += 1



class CUDACheck(object):
    log = "cuda_profile_0.log"
    def __init__(self, config):
        if config.args.cuda_profile:
            log.info("setting CUDA_PROFILE envvar, will write logs such as %s " % self.log )
            os.environ['CUDA_PROFILE'] = "1"
        self.config = config
        self.kernel = config.args.kernel
        self.parser = Parser()
        self.profile = []

    def parse_profile(self):
        self.profile = []
        if self.config.args.cuda_profile:
            for d in self.parser(self.log):
                #print d
                if d['method'].startswith(self.kernel):
                    self.profile.append(d)
    
    def compare_with_launch_times(self, times, launch):
        nprofile = len(self.profile)
        nlaunch = len(times)
        log.info("nprofile %s nlaunch %s " % (nprofile, nlaunch))
        if nprofile > nlaunch:
            profile = self.profile[-nlaunch:]
        elif nprofile == nlaunch:
            profile = self.profile
        else:
            log.info("times : %s " % repr(times))
            return  

        assert len(profile) == nlaunch, (len(profile), nlaunch)
        anno = ["%15.5f" % t + " %(gputime)15.1f %(cputime)15.3fs %(occupancy)s " % d for d,t in zip(profile, times)]
        print launch.annotate(anno)



class Launch(object):
    def __init__(self):
        pass
    def resize(self, size):
        self.work = size

    def annotate(self, anno=[]):
        """
        :param anno: list of per-launch annotation information, eg CUDA_PROFILE results or pycuda timings
        """
        present = self.present
        assert len(anno) == len(present)
        return "\n".join([repr(self)] +["%s : %s" % (l,a) for l, a in  zip(present, anno)])

    def check_counts(self):
        counts = self.counts
        assert sum(counts) == self.total

    counts = property(lambda self:[_[1] for _ in self.iterator])
    total = property(lambda self:reduce(mul,self.work,1))
    __str__ = lambda self:"\n".join(self.present)



class Launch1D(Launch):
    def __init__(self, size, max_blocks=1024, threads_per_block=64 ):
        """
        :param size: 1/2/3-dimensional tuple/array with work size eg (1024,768) 
        """
        self.work = size
        self.max_blocks = max_blocks
        self.threads_per_block = threads_per_block
        self.block = (threads_per_block,1,1)
        Launch.__init__(self)

    iterator = property(lambda self:chunk_iterator(self.total, self.threads_per_block, self.max_blocks))

    def _present(self):
        def present_launch((offset, count, blocks_per_grid)):
            grid=(blocks_per_grid, 1)
            return "offset %10s count %s grid %s block %s " % ( offset, count, repr(grid), repr(self.block) )
        return map(present_launch, self.iterator)
    present = property(_present) 

    __repr__ = lambda self:"%s work %s total %s max_blocks %s threads_per_block %s block %s " % (self.__class__.__name__, self.work, self.total, \
                     self.max_blocks, self.threads_per_block, repr(self.block) )


class Launch2D(Launch):
    def __init__(self, work=(1024,768,1), launch=(2,3,1), block=(16,16,1) ):
        self.work = work
        self.launch = launch
        self.block = block
        Launch.__init__(self)

    iterator = property(lambda self:launch_iterator_2d(self.work, self.launch, self.block))

    def reconfig(self, **kwa):
        for qty in ('launch','block',):
            if qty in kwa:
                setattr(self, qty, kwa[qty])
            pass 
        pass
        # work can be changed with resize

    def _present(self):
        def present_launch( (launch_index, work_count, offset, grid, block)):
            return "launch_index %s work_count %s offset %-20s " % ( launch_index, work_count, offset,  ) + "    ".join(["grid",rep_(grid), "block",rep_(block)])
        return map(present_launch, self.iterator)
    present = property(_present) 

    __repr__ = lambda self:" ".join(["%s %s" % _ for _ in zip("wrk lch blk".split(),map(rep_,(self.work,self.launch,self.block)))])



class Config(object):
    def __init__(self, doc):
        parser, defaults = self._make_parser(doc)
        self.defaults = defaults
        self.args = parser.parse_args()
 
    def _make_parser(self, doc):
        parser = argparse.ArgumentParser(doc)

        defaults = OrderedDict()
        defaults['threads_per_block'] = os.environ.get("THREADS_PER_BLOCK", 64)
        defaults['max_blocks'] = os.environ.get("MAX_BLOCKS", 1024)        # larger max_blocks reduces the number of separate launches, and increasing launch time (BEWARE TIMEOUT)
        defaults['dimension'] = 1

        defaults['block'] = "16,16,1"
        defaults['launch'] = "2,3,1"

        parser.add_argument( "worksize", nargs='+',  help="One or more integers the product of which is the total worksize", type=int  )
        parser.add_argument( "-t","--threads-per-block", help="", type=int  )
        parser.add_argument( "-b","--max-blocks", help="", type=int  )
        parser.add_argument( "-d","--dimension", help="Number of dimensions, 1 or 2", type=int  )

        parser.add_argument( "--block", help="USED FOR 2D ONLY : String 3-tuple dimensions of the block of CUDA threads, eg \"32,32,1\" \"16,16,1\" \"8,8,1\" ", type=str  )
        parser.add_argument( "--launch", help="USED FOR 2D ONLY : String 3-tuple dimensions of the sequence of CUDA kernel launches, eg \"1,1,1\",  \"2,2,1\", \"2,3,1\" ", type=str  )

        parser.set_defaults(**defaults)
        return parser, defaults


    def _settings(self, args, defaults):
        wid = 20
        fmt = " %-15s : %20s : %20s "
        return "\n".join([ fmt % (k,str(v)[:wid],str(getattr(args,k))[:wid]) for k,v in defaults.items() ])

    def __repr__(self):
        return self._settings( self.args, self.defaults )
 
    block=property(lambda self:ivec_(self.args.block))
    launch=property(lambda self:ivec_(self.args.launch))



def main():
    config = Config(__doc__)

    if config.args.dimension == 1:
        launch = Launch1D(config.args.worksize, max_blocks=config.args.max_blocks, threads_per_block=config.args.threads_per_block)
    elif config.args.dimension == 2:
        launch = Launch2D(config.args.worksize, launch=config.launch, block=config.block )
    else:
        assert 0   

    print repr(launch)
    print str(launch)


def test_2d():
    work_a = (1024,768)
    work_b = (1440,852)
    work_c = (300,200)

    work_list = [work_a,work_b,work_c] 
    #work_list = np.random.randint(300,2000,(100,2))

    for work in work_list:
        launch = Launch2D( work, launch=(2,3,1), block=(16,16,1))
        print "\n",repr(launch), "\n", launch


if __name__ == '__main__':
    #main()
    launch = Launch2D()
    print "\n".join(launch.present)
    print "\n".join(launch.present)

