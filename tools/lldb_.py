
import os, sys

DIR = "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/LLDB.framework/Resources/Python"
if os.path.isdir(DIR):
    sys.path.append(DIR)
pass
import lldb


