#!/usr/bin/env bash
# Install the mamba-ssm / causal-conv1d CUDA kernels for the GPU of THIS runtime
# (T4 = sm_75, L4 = sm_89, A100 = sm_80, H100 = sm_90), with a wheel cache on Drive.
#
#   bash scripts/install_mamba.sh            # uses $D/wheels/sm_XX as cache
#
# Order of attempts (stops at the first that passes the GPU self-test):
#   1. wheels cached on Drive for this GPU architecture (seconds);
#   2. pip install with the upstream prebuilt wheels (about 1 min, needs a
#      torch / CUDA combination for which upstream published a wheel);
#   3. build from the PyPI source, compiled ONLY for this architecture
#      (about 15-25 min on T4/A100 instead of hours for all architectures).
# Whatever succeeds is saved to the Drive cache so the next session is fast.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=${D:-/content/drive/MyDrive/DS-MambaGCN_runs}
CC=$(python -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'{m}{n}')")
GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
CACHE="$D/wheels/sm_$CC"
mkdir -p "$CACHE"
echo "GPU: $GPU (sm_$CC), wheel cache: $CACHE"
pip install -q einops ninja packaging

selftest() {
  python - <<'EOF'
import torch
from mamba_ssm import Mamba
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=1).cuda()
x = torch.randn(2, 32, 64, device="cuda", requires_grad=True)
m(x).sum().backward()
torch.cuda.synchronize()
print("mamba_ssm GPU self-test OK")
EOF
}

# the T4 wheels of the first sessions live directly in $D/wheels
if [ "$CC" = "75" ] && ! ls "$CACHE"/*.whl >/dev/null 2>&1 && ls "$D"/wheels/*.whl >/dev/null 2>&1; then
  cp "$D"/wheels/*.whl "$CACHE"/
fi

# 1. Drive cache
if ls "$CACHE"/*.whl >/dev/null 2>&1; then
  pip install -q --no-deps --force-reinstall "$CACHE"/*.whl && selftest && exit 0
  echo "cached wheels do not work with this runtime (torch changed?); rebuilding"
  rm -f "$CACHE"/*.whl
fi

# 2. upstream prebuilt wheels (setup.py downloads a matching wheel from GitHub releases)
W=/content/wheel_build; rm -rf "$W"; mkdir -p "$W"
if pip wheel -q --no-deps --no-build-isolation -w "$W" causal-conv1d mamba-ssm 2>"$W/prebuilt.err" \
   && pip install -q --no-deps --force-reinstall "$W"/*.whl && selftest; then
  cp "$W"/*.whl "$CACHE"/ && echo "saved prebuilt wheels to $CACHE"; exit 0
fi
echo "no usable prebuilt wheel; compiling for sm_$CC only (15-25 min)"

# 3. source build restricted to this architecture
rm -rf "$W"; mkdir -p "$W/src"; cd "$W/src"
python - <<'EOF'
import json, tarfile, urllib.request
for p in ("causal-conv1d", "mamba-ssm"):
    j = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{p}/json"))
    f = next(f for f in j["urls"] if f["packagetype"] == "sdist")
    urllib.request.urlretrieve(f["url"], f["filename"])
    tarfile.open(f["filename"]).extractall(".")
    print("downloaded", f["filename"])
EOF
python "$HERE/patch_gencode.py" "$CC" */setup.py || exit 1
RAMJ=$(( $(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo) / 4 )); J=$(nproc); [ "$RAMJ" -lt "$J" ] && J=$RAMJ; [ "$J" -lt 1 ] && J=1
export MAX_JOBS=${MAX_JOBS:-$J} CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
for p in causal_conv1d mamba_ssm; do
  d=$(ls -d ${p}-*/ | head -1)
  (cd "$d" && pip wheel -q --no-deps --no-build-isolation -w "$W" .) || { echo "build of $p failed"; exit 1; }
  pip install -q --no-deps --force-reinstall "$W"/${p}-*.whl
done
selftest || exit 1
cp "$W"/*.whl "$CACHE"/ && echo "saved compiled wheels to $CACHE"
