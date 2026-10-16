"""Restrict the nvcc -gencode targets in a CUDA extension's setup.py to ONE architecture.

    python scripts/patch_gencode.py 80 causal_conv1d-*/setup.py mamba_ssm-*/setup.py

Upstream mamba-ssm / causal-conv1d compile for every GPU generation, which takes
hours on a Colab VM. Keeping only the GPU of the runtime cuts that to minutes.
Handles the three spellings used by upstream versions:
  cc_flag.append("-gencode")  +  cc_flag.append("arch=compute_80,code=sm_80")
  "-gencode", "arch=compute_80,code=sm_80"          (inside a list)
  "-gencode=arch=compute_80,code=sm_80" / "--generate-code=..."   (single token)
If the target architecture is not in the upstream list (e.g. sm_89), the first
pair is rewritten to the target instead of being removed.
"""
import re
import sys

ARCH = r'arch=compute_(\d+)[a-z]?,code=(?:sm|compute)_\d+[a-z]?'
PATTERNS = [
    # (regex, replacement builder for a REMOVED pair, builder for a RETARGETED pair)
    (re.compile(r'cc_flag\.append\(\s*["\']-gencode["\']\s*\)(\s*\n\s*)cc_flag\.append\(\s*["\']' + ARCH + r'["\']\s*\)'),
     lambda m: "pass" + m.group(1) + "pass",
     lambda m, t: f'cc_flag.append("-gencode"){m.group(1)}cc_flag.append("arch=compute_{t},code=sm_{t}")'),
    (re.compile(r'["\']-gencode["\']\s*,\s*["\']' + ARCH + r'["\']\s*,?'),
     lambda m: "",
     lambda m, t: f'"-gencode", "arch=compute_{t},code=sm_{t}",'),
    (re.compile(r'["\'](?:-gencode|--generate-code)=' + ARCH + r'["\']\s*,?'),
     lambda m: "",
     lambda m, t: f'"-gencode=arch=compute_{t},code=sm_{t}",'),
]


def patch(text, target):
    present = any(m.group(2 if i == 0 else 1) == target
                  for i, (rx, _, _) in enumerate(PATTERNS) for m in rx.finditer(text))
    stats = {"kept": 0, "removed": 0, "retargeted": 0}
    need_target = [not present]

    for i, (rx, remove, retarget) in enumerate(PATTERNS):
        arch_group = 2 if i == 0 else 1

        def repl(m):
            if m.group(arch_group) == target:
                stats["kept"] += 1
                return m.group(0)
            if need_target[0]:
                need_target[0] = False
                stats["retargeted"] += 1
                return retarget(m, target)
            stats["removed"] += 1
            return remove(m)
        text = rx.sub(repl, text)
    return text, stats


def main():
    target, paths = sys.argv[1], sys.argv[2:]
    if not paths:
        sys.exit("usage: patch_gencode.py <cc, e.g. 80> <setup.py> ...")
    for p in paths:
        src = open(p).read()
        out, st = patch(src, target)
        compile(out, p, "exec")                      # still valid Python
        if st["kept"] + st["retargeted"] == 0:
            sys.exit(f"{p}: no -gencode entry found, upstream layout changed; patch by hand")
        open(p, "w").write(out)
        print(f"{p}: sm_{target} only ({st})")


if __name__ == "__main__":
    main()
