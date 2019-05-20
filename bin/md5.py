#!/usr/bin/env python
try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5
pass
import os, logging
log = logging.getLogger(__name__)


cumulative = None

def digest_(path):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::

        md5 /Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf 
        MD5 (/Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf) = 3a63b5232ff7cb6fa6a7c241050ceeed

    And md5sum on linux:: 

        [blyth@localhost opticks]$ md5.py /tmp/blyth/location/test_array_digest.npy
        79da1243f9e602c699ec9df8527acdf7
        [blyth@localhost opticks]$ md5sum /tmp/blyth/location/test_array_digest.npy
        79da1243f9e602c699ec9df8527acdf7  /tmp/blyth/location/test_array_digest.npy

    """
    global cumulative

    if not os.path.exists(path):return None
    if os.path.isdir(path):return None
    dig = md5()

    if cumulative is None:
        cumulative = md5() 

    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(8192),''): 
            dig.update(chunk)
            cumulative.update(chunk)
        pass
    return dig.hexdigest()



if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    print digest_(sys.argv[1])

