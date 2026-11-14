#!/bin/bash
# GPU-free gate-3 check of the re-cut 903 on b12177 (the v0.40.16 fold), as mmq-compile-0413.sh: mmq.cu compiles
# with the guard for sm_75 and sm_120a, and a mutation (helper without the nthreads padding) fails the build on the
# static_assert, naming tables, with no syntax errors. Each gencode drives its own config table (memory: gpu-free
# checks). The tree is b12177 + 805 + 903, unpacked from git archive (A/clean-check).
set -u
A=${A:-/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch/assess-0416}; T=$A/clean-check; NVCC=/usr/local/cuda-13.0/bin/nvcc
echo "### MMQ-COMPILE-0416 start $(date '+%F %T')"
cd $T || exit 1
cp ggml/src/ggml-cuda/mmq.cuh $A/mmq.cuh.pristine
for arch in 75 120a; do
  B=$A/build-mmq-$arch
  [ -f $B/build.ninja ] || nice cmake -S $T -B $B -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$NVCC -DCMAKE_CUDA_ARCHITECTURES=$arch \
     -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DCMAKE_BUILD_TYPE=Release > $B.configure.log 2>&1 \
     || { echo "### configure $arch FAILED: $(tail -3 $B.configure.log)"; continue; }
  obj=$(cd $B && ninja -t targets all | grep -oE '^[^:]*mmq\.cu\.o' | head -1)
  ( cd $B && nice ninja "$obj" > $B.mmq.log 2>&1 ); rc=$?
  echo "### sm_$arch compile rc=$rc ($(grep -c ': error' $B.mmq.log) errors) $(date '+%T')"
  # mutation: the helper loses its padding to nthreads ints
  python3 - <<'PY'
p='ggml/src/ggml-cuda/mmq.cuh'; s=open(p).read()
old='    return GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));'
assert s.count(old)==1; open(p,'w').write(s.replace(old,'    return config.J*sizeof(block_q8_1_mmq);'))
PY
  ( cd $B && NVCC_APPEND_FLAGS="-Xcudafe --error_limit=100000" nice ninja "$obj" > $B.mut.log 2>&1 ); mrc=$?
  cp $A/mmq.cuh.pristine ggml/src/ggml-cuda/mmq.cuh
  sa=$(grep -c 'static assertion failed\|static_assert' $B.mut.log); syn=$(grep -cE 'error: expected|closing brace|identifier .* is undefined' $B.mut.log)
  tabs=$(grep -oE 'config in the [a-z0-9_]+ table' $B.mut.log | sort -u | sed 's/config in the //;s/ table//' | tr '\n' ' ')
  echo "### sm_$arch mutation rc=$mrc static_assert lines=$sa syntax errors=$syn tables: $tabs"
done
cmp ggml/src/ggml-cuda/mmq.cuh $A/mmq.cuh.pristine && echo "### mmq.cuh restored"
echo "### MMQ-COMPILE-0416 DONE $(date '+%F %T')"
