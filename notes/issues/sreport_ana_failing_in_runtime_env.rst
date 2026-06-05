sreport_ana_failing_in_runtime_env
====================================


FIXED : opticks.sysrap.tests py module missing in runtime env
----------------------------------------------------------------


::

    [lu] A[blyth@localhost opticks]$ A=A9 B=B9 ~/o/sreport_ab.sh ana
    Python 3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:02) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.1.0 -- An enhanced Interactive Python. Type '?' for help.
    Tip: You can find how to type a Unicode symbol by back-completing it, eg `\Ⅷ<tab>` will expand to `\ROMAN NUMERAL EIGHT`.
    pvplt MODE:2
    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    File ~/opticks/sysrap/tests/sreport_ab.py:6
          4 from opticks.ana.fold import Fold
          5 from opticks.ana.npmeta import NPMeta
    ----> 6 from opticks.sysrap.tests.sreport import Substamp
          7 from opticks.sysrap.tests.sreport import RUN_META
          9 MODE = 2

    ModuleNotFoundError: No module named 'opticks.sysrap.tests'
    > /home/blyth/opticks/sysrap/tests/sreport_ab.py(6)<module>()
          4 from opticks.ana.fold import Fold
          5 from opticks.ana.npmeta import NPMeta
    ----> 6 from opticks.sysrap.tests.sreport import Substamp
          7 from opticks.sysrap.tests.sreport import RUN_META
          8

    ipdb>

Kludge mixing trees works when have source::

    PYTHONPATH=$HOME  A=A9 B=B9 ~/o/sreport_ab.sh ana


After fix can::

    A=A9 B=B9 sreport_ab.sh ana



