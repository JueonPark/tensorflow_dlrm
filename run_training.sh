#!/bin/bash
shopt -s extglob

for filename in xla_hlo/*; do
  if [[ $filename == *"offline_execution_result"* ]]
  then
    echo $filename
    continue
  else
    rm $filename
  fi
done
rm xla_hlo/*/*

# Basic Configurations
BATCH=1024
FACTOR=64
EPOCHS=1

# XLA
export TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2"
export TF_DUMP_GRAPH_PREFIX="./xla_hlo"

# simulation
export TRACER_TOOL=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/tracer_tool.so
export POST_PROCESSING=/home/jueonpark/cxl-simulator/multi_gpu_simulator/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing

# caution!
# 1939
export DYNAMIC_KERNEL_LIMIT_START=999998
export DYNAMIC_KERNEL_LIMIT_END=999999

# additional runtime environment variables for tensorflow
export TF_CPP_MIN_VLOG_LEVEL=1
# export ENABLE_CONSOLE=true

# EXECUTION OPTION
TRACE=${1:-notrace}
EXEC=${2:-help}
OVERLAP=${3:-0} # 0:one-by-one, 1:one_ndpx-by-many_gpu, 2:one_gpu-by-many_ndpx
OFFLINE=${4:-0}

if [ $EXEC = "vanilla" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text  --xla_dump_to=./xla_hlo "
elif [ $EXEC = "ndpx" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_op_offloading=true --xla_ndpx_use_overlapping_strategy=$OVERLAP --xla_ndpx_use_offline_result=$OFFLINE  --xla_dump_to=./xla_hlo "
else
  echo "trace flags: trace, notrace"
  echo "execution flags: vanilla, ndpx"
  echo "overlap flags: 0(one-by-one), 1(one_ndpx-by-many_gpu), 2(one_gpu-by-many_ndpx)"
  echo "offine result flags: 0(real gpu), 1(simulation)"
  exit 0
fi

if [ $TRACE = "trace" ]
then
  LD_PRELOAD=$TRACER_TOOL python train.py 1> output 2> error
  $POST_PROCESSING ./traces/kernelslist
else
  python train.py 1> output 2> error
fi
