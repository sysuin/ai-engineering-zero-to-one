# A test is a function that states what should be true, and pytest runs every one.
# This listing runs the tests in _test_pricing.py and shows what pytest prints.

import os
import sys

import pytest

os.environ["COLUMNS"] = "80"           # so pytest's banner lines fit the page
sys.path.insert(0, "code/02")
code = pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:warnings", "-rN",
                    "--tb=short", "code/02/_test_pricing.py"])
print(f"\npytest exit code: {int(code)}  (0 means every test passed)")
