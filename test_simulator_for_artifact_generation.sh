#!/usr/bin/env bash

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"

TO_TEST="scatter"
IN_DIR="$ROOT_DIR/results/example_for artifact_simulation/simulation_001/"
OUT_DIR="$ROOT_DIR/results/test_scatter_artifacts/"
SCANNER="default"
ZSLICE=59

while getopts t:i:o:s:z:h flag
do
    case "${flag}" in
        t) TO_TEST=${OPTARG};;
        i) IN_DIR=${OPTARG};;
        o) OUT_DIR=${OPTARG};;
        s) SCANNER=${OPTARG};;
        z) ZSLICE=${OPTARG};;
        h)
           echo "Run DEBISim simulation for a CT Phantom (ACR | Battelle A/B) "
           echo
           echo "Syntax: ./simulate_ct_phantom.sh [-t|i|o|s|z] "
           echo "options:"
           echo "t      artifacts to test (ring | motion | scatter | bag) - bag tests baggage creation artifacts"
           echo "i      input simulation directory"
           echo "o      output directory"
           echo "s      scanner (sensation_32 | definition_as | force | default)"
           echo "z      ct z slice to process"
           echo
           exit
           ;;
    esac
done

echo "$ROOT_DIR"
echo "$PYTHONPATH"
echo "Running CT simulation to test simulation of ($TO_TEST) artifacts for scanner ($SCANNER) at: $OUT_DIR from $IN_DIR"

case $TO_TEST in
    ring)
        python scripts/test_ring_artifact_generation.py --out_dir=$OUT_DIR --scanner=$SCANNER --in_dir=$IN_DIR --zslice=$ZSLICE
        ;;
    motion)
        python scripts/test_motion_artifact_generation.py --out_dir=$OUT_DIR --scanner=$SCANNER
        ;;
    scatter)
        python scripts/test_scatter_artifact_generation.py --out_dir=$OUT_DIR --scanner=$SCANNER --in_dir=$IN_DIR --zslice=$ZSLICE
        ;;
    bag)
        python scripts/test_baggage_creator_performance.py --out_dir=$OUT_DIR --scanner=$SCANNER
        ;;
    *)
        echo "Test not recognized - choose from {acr | battelle_a | battelle_b}"
        ;;
esac