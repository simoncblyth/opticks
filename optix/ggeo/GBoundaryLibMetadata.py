#!/usr/bin/env python

import sys, logging, json, os
log = logging.getLogger(__name__)


def dump(idpath, sub=None):
    path = os.path.join(idpath, 'GBoundaryLibMetadata.json')

    with open(path, 'r') as fp:
        js = json.load(fp)

    if sub is None:
        isubs = sorted(map(int, js['lib']['boundary'].keys() ))
        print isubs
        for isub in isubs:
            sub = js['lib']['boundary'][str(isub)]
            print " %2s : %2s :   %25s %25s %25s %25s " % ( isub, isub+1, sub['imat']['shortname'], sub['omat']['shortname'], sub['isur']['name'], sub['osur']['name'] )
        pass
    else:
        sub = js['lib']['boundary'][str(sub)]

        print sub['imat'].keys() 


def main():
    logging.basicConfig(level=logging.INFO)
    log.info(sys.argv)

    idpath = os.environ['IDPATH']
    sub = sys.argv[1] if len(sys.argv)>1 else None

    dump(idpath, sub)


if __name__ == '__main__':
    main()

