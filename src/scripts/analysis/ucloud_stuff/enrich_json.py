#!/usr/bin/env python
from json import dumps, loads
from sys import stderr, stdin, stdout

if __name__ == "__main__":
    for line in stdin.readlines():
        d = loads(line)
        model = d["model"]
        parts = model.split("-")
#        print(d, model, parts, file=stderr)
        d["num_metrics"] = len(d)-1
        d["experts"] = ["public", "code", "creative", "math", "news", "pes2o", "reddit"] if model.startswith("FlexOlmo") else ["public", parts[1]]
        d["num_experts"] = int(model.split("x7B")[0].split("-")[-1]) if "x7B" in model else 1
        d["active_experts"] = int(model.split("-a")[1].split("-")[0]) if "-a" in model else (2 if "x7B" in model else 1)
        d["rank"] = int(parts[-1][1:]) if parts[-1].startswith("r") else 2**(17 if parts[-1] == "best2" else (16 if parts[-1] == "best" else 15))
        print(dumps(d))