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
export DYNAMIC_KERNEL_LIMIT_START=999998
export DYNAMIC_KERNEL_LIMIT_END=999999

# additional runtime environment variables for tensorflow
# export TF_CPP_MIN_VLOG_LEVEL=1
# export ENABLE_CONSOLE=true

# execution options:
# $1:
# - vanila for no
# - pm for pattern matching
# - fo for fusion offloading
# - pm_fo for both pattern matching & fusion offlaoding
# - ideal for ideal offloading
# - pm_ideal for both pattern matching & ideal offloading
# $2: trace generation
# - keyword "trace" given
# $3: xla_ndpx_use_offline_result
# - 0 for using GPU results
# - 1 for using SIM results
# - on default(no $3 input), use GPU results
if [ $1 = "vanila" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text  --xla_dump_to=./xla_hlo "
elif [ $1 = "pm" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_to=./xla_hlo "
elif [ $1 = "fo" ]
then
  if [ $# = 3 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=$3 --xla_dump_to=./xla_hlo "
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=0 --xla_dump_to=./xla_hlo "
  fi
elif [ $1 = "pm_fo" ]
then
  if [ $# = 3 ]
  then
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=$3 --xla_dump_to=./xla_hlo "
  else
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_offline_result=0 --xla_dump_to=./xla_hlo "
  fi
elif [ $1 = "ideal" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_ideal_offloading --xla_dump_to=./xla_hlo "
elif [ $1 = "pm_ideal" ]
then
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_gpu_use_cudnn_batchnorm=true --xla_gpu_use_ndp_batchnorm=true --xla_gpu_use_ndp_bert_pattern=true --xla_gpu_use_ndp_optimizer=true --xla_dump_hlo_as_text --xla_ndpx_use_fusion_offloading=true --xla_ndpx_use_ideal_offloading --xla_dump_to=./xla_hlo "
else
  echo "flags: vanila, pm, fo, pm_fo, ideal, idea_fo"
	exit 0
fi

# whether to get trace or not
if [ $# -ge 2 ] && [ $2 = "trace" ]
then
  LD_PRELOAD=$TRACER_TOOL python train.py
  $POST_PROCESSING ./traces/kernelslist
else
  python train.py
fi
