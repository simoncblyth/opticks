#!/usr/bin/env python
"""
py3check.py
============

Running this finds all .py files recursively 
within the invoking directory and attempts to parse 
the code with the invoking python interpreter 
without actually running the scripts/modules.

::

   cd ~/opticks
   py3check.py 
   ...
   pymajor:3 tot:346 tot_pass:244 tot_fail:102 frac_pass: 0.71  frac_fail: 0.29 


Despite the name, this also checks py2 when invoked with that version::

    epsilon:opticks blyth$ ip2
    ip2 () 
    { 
        PYMAJOR=2 source ~/.python_config;
        ipython $*
    }

    In [1]: run bin/py3check.py 
    pymajor:2 tot:346 tot_pass:346 tot_fail:0 frac_pass: 1.00  frac_fail: 0.00 



"""
import sys, os, ast, fnmatch, logging
log = logging.getLogger(__name__)

def ast_parse(path):
    """
    * :google:`check python3 compat without running the script`
    * https://stackoverflow.com/questions/40886456/how-to-detect-if-code-is-python-3-compatible
    """
    code = open(path, 'rb').read()
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    pymajor = sys.version_info.major 
    tot=0
    tot_pass=0  
    tot_fail=0  
  
    for dirpath, dirnames, names in os.walk("."):
        for name in fnmatch.filter(names, "*.py"):
            path = os.path.join(dirpath, name) 
            ok = ast_parse(path)
            tot += 1 
            if ok:
                tot_pass += 1 
            else:
                tot_fail += 1 
            pass
            st = "Y" if ok else "N"
            if st == "N":
                print("[%s] %s " %  (st, path))
            pass
        pass
    pass

    frac_pass = float(tot_pass)/float(tot)
    frac_fail = float(tot_fail)/float(tot)
    msg = "pymajor:{pymajor} tot:{tot} tot_pass:{tot_pass} tot_fail:{tot_fail} frac_pass:{frac_pass:5.2f}  frac_fail:{frac_fail:5.2f} ".format(**locals())
    print(msg)



