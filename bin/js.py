#!/usr/bin/env python

import sys, json, os, logging, pprint
log = logging.getLogger(__name__)

js = json.load(file(sys.argv[1]))

pprint.pprint(js) 


