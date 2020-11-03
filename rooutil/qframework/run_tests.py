#!/usr/bin/env python

import imp
import sys
import inspect
from glob import glob
import re
import unittest


# Main suite, which holds all TestCases
main_suite = unittest.TestSuite()


# Loop over all files in the test directory
for pyfile in glob("test/*.py"):

    # Extract the module name
    m = re.match("test/(.*)\\.py", pyfile)
    if not m:
        continue
    test_case_name = m.group(1)

    # Import module
    test_module = imp.load_source("%s" % test_case_name, pyfile)

    # Loop over items in the module
    for name, item in vars(test_module).items():
        
        # Skip items, which are not TestCase
        if not (inspect.isclass(item) and issubclass(item, unittest.TestCase)):
            continue

        # Build suite from this TestCase and add it to the main suite
        single_suite = unittest.TestLoader().loadTestsFromTestCase(item)
        main_suite.addTest(single_suite)


# Run Tests
result = unittest.TextTestRunner(verbosity=3).run(main_suite)

# Check result
if not result.wasSuccessful():
    sys.exit(1)
