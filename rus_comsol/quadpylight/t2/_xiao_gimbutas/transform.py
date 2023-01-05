import argparse
import json
import re

import fjson

parser = argparse.ArgumentParser(description="Transform to new format.")
parser.add_argument("filename", type=str)
args = parser.parse_args()

with open(args.filename) as f:
    content = json.load(f)

data = {}
for key in ["centroid", "d3_aa", "d3_ab", "c3_ab"]:
    if key in content:
        data[key] = content[key]

degree = content["degree"]
m = re.match("xg(..)\\.json", args.filename)

idx = m.group(1)
if idx[0] == "0":
    idx = idx[1]


name = f"Xiao-Gimbutas {idx}"

c = {
    "name": name,
    "domain": "T2",
    "degree": degree,
    "test_tolerance": 1.0e-100,
    "data": {key: list(map(list, zip(*value))) for key, value in data.items()},
}


with open(args.filename, "w") as f:
    fjson.dump(c, f, indent=2, float_format=".16e")
