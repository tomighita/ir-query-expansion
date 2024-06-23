import pyterrier as pt
import sys

if len(sys.argv) <= 1:
    exit(-1)

print(sys.argv)

if not pt.started():
    pt.init(
        tqdm="notebook",
        boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
    )

for ds in sys.argv[1:]:
    pt.get_dataset(ds)

print("Checked datasets, all good! Moving to indexing...")