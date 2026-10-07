#!/bin/bash
# Diagnostic: SAC-sized update speed under different math-library settings.
grep -m1 "model name" /proc/cpuinfo; grep -o -m1 -w avx2 /proc/cpuinfo
cat > bench.py <<'PY'
import os, time, torch
torch.set_num_threads(1)
net = torch.nn.Sequential(torch.nn.Linear(41, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
opt = torch.optim.Adam(net.parameters()); x = torch.randn(256, 41)
for _ in range(50):
    loss = net(x).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
t = time.time()
for _ in range(500):
    loss = net(x).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
fw = time.time(); a = torch.randn(256, 256); b = torch.randn(256, 256)
for _ in range(2000): a @ b
print(f"{os.environ.get('LABEL','default'):28s} updates/s {500/(fw-t):5.0f}   matmul256/s {2000/(time.time()-fw):6.0f}")
PY
python -c "import torch; print(torch.__config__.show())" | grep -iE "blas|mkl|lapack|cpu capability" | head -5
LABEL=default python bench.py
LABEL=MKL_ENABLE_INSTRUCTIONS=AVX2 MKL_ENABLE_INSTRUCTIONS=AVX2 python bench.py
LABEL=MKL_DEBUG_CPU_TYPE=5 MKL_DEBUG_CPU_TYPE=5 python bench.py
LABEL=ATEN_CPU_CAPABILITY=avx2 ATEN_CPU_CAPABILITY=avx2 python bench.py
LABEL=mkldnn_off python -c "import torch; torch.backends.mkldnn.enabled=False; exec(open('bench.py').read())"
LABEL=flush_denormal python -c "import torch; torch.set_flush_denormal(True); exec(open('bench.py').read())"
