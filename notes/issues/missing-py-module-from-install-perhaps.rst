missing-py-module-from-install-perhaps.rst
============================================


Clean install revealed some python modules missing from install::

    om-prefix-clean
    opticks-full


::

    Python 3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:02) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.1.0 -- An enhanced Interactive Python. Type '?' for help.
    Tip: Use `object?` to see the help on `object`, `object??` to view its source
    pvplt MODE:3 
    [from opticks.ana.p import * 
    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    File ~/j/InputPhotonsCheck/InputPhotonsCheck.py:12
          9 log = logging.getLogger(__name__)
         10 os.environ["MODE"] = "3"
    ---> 12 from opticks.sysrap.sevt import SEvt, SAB, SABHit
         13 from opticks.ana.p import cf
         15 MODE = int(os.environ.get("MODE","3"))

    File /data1/blyth/local/opticks_Debug/py/opticks/sysrap/sevt.py:14
         11 from opticks.ana.fold import Fold
         13 print("[from opticks.ana.p import * ")
    ---> 14 from opticks.ana.p import *
         15 print("]from opticks.ana.p import * ")
         17 from opticks.ana.eget import eslice_

    File /data1/blyth/local/opticks_Debug/py/opticks/ana/p.py:43
         41 import numpy as np
         42 import hashlib, builtins
    ---> 43 from opticks.ana.hismask import HisMask   
         44 from opticks.ana.histype import HisType  
         45 from opticks.ana.nibble import count_nibbles 

    File /data1/blyth/local/opticks_Debug/py/opticks/ana/hismask.py:42
         39 import numpy as np
         41 from opticks.ana.base import PhotonMaskFlags
    ---> 42 from opticks.ana.seq import MaskType, SeqTable, SeqAna
         43 from opticks.ana.nbase import count_unique_sorted
         44 from opticks.ana.nload import A

    ModuleNotFoundError: No module named 'opticks.ana.seq'
    > /data1/blyth/local/opticks_Debug/py/opticks/ana/hismask.py(42)<module>()
         40 
         41 from opticks.ana.base import PhotonMaskFlags
    ---> 42 from opticks.ana.seq import MaskType, SeqTable, SeqAna
         43 from opticks.ana.nbase import count_unique_sorted
         44 from opticks.ana.nload import A
    ipdb>

    In [1]: print("\n".join(os.environ.get("PYTHONPATH").split(":")))
    /home/blyth/junosw/InstallArea/python
    /home/blyth/junosw/InstallArea/lib64
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/junosw/InstallArea/python
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/junosw/InstallArea/lib64
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/mt.sniper/InstallArea/lib64
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/mt.sniper/InstallArea/python
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/sniper/InstallArea/lib64
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/sniper/InstallArea/python
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/ROOT/6.30.08/lib
    /data1/blyth/local/opticks_Debug/py



    In [2]: import opticks

    In [3]: opticks.__file__
    Out[3]: '/data1/blyth/local/opticks_Debug/py/opticks/__init__.py'



    In [2]: import opticks

    In [3]: opticks.__file__
    Out[3]: '/data1/blyth/local/opticks_Debug/py/opticks/__init__.py'

    In [4]: import opticks.ana

    In [5]: import opticks.ana.seq
    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    Cell In[5], line 1
    ----> 1 import opticks.ana.seq

    ModuleNotFoundError: No module named 'opticks.ana.seq'
    > <ipython-input-5-8a25d9e2b931>(1)<module>()
    ----> 1 import opticks.ana.seq

    ipdb>

    In [6]: opticks.ana.__file__
    Out[6]: '/data1/blyth/local/opticks_Debug/py/opticks/ana/__init__.py'




