#!/usr/bin/env bash

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"

PHANTOM="acr"
OUT_DIR="$ROOT_DIR/results/simulation_acr/"
SCANNER="sensation_32"

while getopts p:o:s:h flag
do
    case "${flag}" in
        p) PHANTOM=${OPTARG};;
        o) OUT_DIR=${OPTARG};;
        s) SCANNER=${OPTARG};;
        h)
           echo "Run DEBISim simulation for a CT Phantom (ACR | Battelle A/B) "
           echo
           echo "Syntax: ./simulate_ct_phantom.sh [-p|o|s] "
           echo "options:"
           echo "p      phantom (acr | battelle_a | battelle_b)"
           echo "o      output directory"
           echo "s      scanner (sensation_32 | definition_as | force | default)"
           echo
           exit
           ;;
    esac
done

echo "$ROOT_DIR"
echo "$PYTHONPATH"
echo "Running CT simulation for ($PHANTOM) phantom for scanner ($SCANNER) at: $OUT_DIR"

case $PHANTOM in
    acr)
        python scripts/create_acr_phantom_simulation.py --out_dir=$OUT_DIR --scanner=$SCANNER
        ;;
    battelle_a)
        python scripts/create_battelle_phantom_simulation.py --out_dir=$OUT_DIR --scanner=$SCANNER --ptype='a'
        ;;
    battelle_b)
        python scripts/create_battelle_phantom_simulation.py --out_dir=$OUT_DIR --scanner=$SCANNER --ptype='b'
        ;;
    *)
        echo "Phantom not recognized - choose from {acr | battelle_a | battelle_b}"
        ;;
esac