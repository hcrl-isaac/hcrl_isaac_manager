"""Run hcrl_isaaclab's artifact resolver against a dest inside a staged tree (helper for test_stage.sh).

Usage: resolve_in_tree.py <artifacts.py> <tree resources dir> <rel_path> <new version dir>
"""

import importlib.util
import os
import shutil
import sys

src, resources, rel_path, new_dir = sys.argv[1:5]
dst = os.path.join(resources, "hcrl_isaaclab", "hcrl_isaaclab", "utils", "artifacts.py")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)  # RESOURCES_DIR is derived from the module's location, so it must sit in the tree
spec = importlib.util.spec_from_file_location("tree_artifacts", dst)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
assert os.path.abspath(resources) == mod.RESOURCES_DIR, mod.RESOURCES_DIR
mod._download = lambda _spec: new_dir
mod._fetch(mod.ResourceSpec(key="k", rel_path=rel_path, version="v1"))
print(os.readlink(os.path.join(resources, rel_path)))
