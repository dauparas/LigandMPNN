#!/bin/bash

script_dir=$(dirname $0)
cd $script_dir
set -e 

python scripts/run.py \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/autoregressive_score_w_seq" \
        scorer.use_sequence=True \
        sampling.batch_size=1 \
        sampling.number_of_batches=10

python scripts/run.py \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/autoregressive_score_wo_seq" \
        scorer.use_sequence=False \
        sampling.batch_size=1 \
        sampling.number_of_batches=10


python scripts/run.py \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.single_aa_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/single_aa_score_w_seq" \
        scorer.use_sequence=True \
        sampling.batch_size=1 \
        sampling.number_of_batches=10


python scripts/run.py \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.single_aa_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/single_aa_score_wo_seq" \
        scorer.use_sequence=False \
        sampling.batch_size=1 \
        sampling.number_of_batches=10