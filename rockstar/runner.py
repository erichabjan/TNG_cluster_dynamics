import subprocess
import os
import numpy as np

clusters = np.array([98])
#clusters = np.arange(99)
procs = []

for cid in clusters:
    argv = [
        "python3",
        "/home/habjan.e/TNG/TNG_cluster_dynamics/rockstar/sub_finder.py",
        str(cid),
    ]
    procs.append(subprocess.Popen(argv))

for p in procs:
    p.wait()

print(f'ROCKSTAR ran for clusters: {clusters}')