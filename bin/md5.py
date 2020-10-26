#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#
"""
md5.py
========


Beware newlines::

    epsilon:bin blyth$ echo hello > /tmp/hello
    epsilon:bin blyth$ xxd /tmp/hello
    00000000: 6865 6c6c 6f0a                           hello.
    epsilon:bin blyth$ md5 /tmp/hello
    MD5 (/tmp/hello) = b1946ac92492d2347c6235b4d2611184


    epsilon:bin blyth$ echo -n hello > /tmp/hello-n
    epsilon:bin blyth$ xxd /tmp/hello-n
    00000000: 6865 6c6c 6f                             hello

    epsilon:bin blyth$ md5 /tmp/hello-n
    MD5 (/tmp/hello-n) = 5d41402abc4b2a76b9719d911017c592
    epsilon:bin blyth$ md5 -s hello
    MD5 ("hello") = 5d41402abc4b2a76b9719d911017c592
    epsilon:bin blyth$ 

    epsilon:bin blyth$ python -c "import sys, codecs, hashlib ; print(hashlib.md5(codecs.latin_1_encode(sys.argv[1])[0]).hexdigest())"  hello
    5d41402abc4b2a76b9719d911017c592



Testing this with py2 and py3::

    epsilon:bin blyth$ md5 -s hello
    MD5 ("hello") = 5d41402abc4b2a76b9719d911017c592

    epsilon:bin blyth$ /opt/local/bin/python ~/opticks/bin/md5.py /tmp/hello 
    b1946ac92492d2347c6235b4d2611184
    epsilon:bin blyth$ /opt/local/bin/python ~/opticks/bin/md5.py /tmp/hello --python-version-override 2
    b1946ac92492d2347c6235b4d2611184
    epsilon:bin blyth$ /opt/local/bin/python ~/opticks/bin/md5.py /tmp/hello --python-version-override 3
    b1946ac92492d2347c6235b4d2611184

    epsilon:bin blyth$ ~/miniconda3/bin/python ~/opticks/bin/md5.py /tmp/hello --python-version-override 2
    Traceback (most recent call last):
      File "/Users/blyth/opticks/bin/md5.py", line 140, in <module>
        print(digest_(path, args.python_version_override))
      File "/Users/blyth/opticks/bin/md5.py", line 122, in digest_
        hexdig = digest2_(path)
      File "/Users/blyth/opticks/bin/md5.py", line 73, in digest2_
        assert pymajor == 2, "this fails with py3" 
    AssertionError: this fails with py3

    epsilon:bin blyth$ ~/miniconda3/bin/python ~/opticks/bin/md5.py /tmp/hello --python-version-override 3
    b1946ac92492d2347c6235b4d2611184
    epsilon:bin blyth$ ~/miniconda3/bin/python ~/opticks/bin/md5.py /tmp/hello
    b1946ac92492d2347c6235b4d2611184
    epsilon:bin blyth$ /opt/local/bin/python ~/opticks/bin/md5.py /tmp/hello 
    b1946ac92492d2347c6235b4d2611184

"""

try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5
pass
import os, sys, logging, argparse
log = logging.getLogger(__name__)


cumulative = None


def digest2_(path, block_size=8192):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path
    """
    global cumulative
    dig = md5()
    if cumulative is None:
        cumulative = md5() 
    pass
    pymajor = sys.version_info[0] 
    assert pymajor == 2, "this hangs with py3" 

    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(block_size),''): 
            dig.update(chunk)
            cumulative.update(chunk)
        pass
    pass
    return dig.hexdigest()


def digest3_(path, block_size=8192):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path
    """
    global cumulative
    dig = md5()
    if cumulative is None:
        cumulative = md5() 
    pass
    with open(path, "rb") as f:
        #while chunk := f.read(block_size):   walrus-operator only available in py38 + it gives error in py27
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            pass
            dig.update(chunk)
            cumulative.update(chunk)
        pass
    pass
    return dig.hexdigest()


def digest_(path, pyver=0):
    """
    :param path:
    :param pyver:  
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::
    """
    hexdig = None
    if pyver == 0 or pyver == 3:
        hexdig = digest3_(path)
    elif pyver == 2:
        hexdig = digest2_(path)
    else:
        assert 0, "invalid pyver %d" % pyver
    pass 
    return hexdig



if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(  "paths", nargs='*', help="File paths to digest" )
    parser.add_argument("--python-version-override", type=int, default=0, choices=[0,2,3], help="Default %(default)s ." )

    args = parser.parse_args()
    for path in args.paths:
        print(digest_(path, args.python_version_override))
    pass

