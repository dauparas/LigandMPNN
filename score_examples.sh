#!/bin/bash

script_dir=$(dirname $0)
cd $script_dir
set -e 

ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/autoregressive_score_w_seq" \
        scorer.use_sequence=True \
        sampling.batch_size=1 \
        sampling.number_of_batches=10

ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/autoregressive_score_wo_seq" \
        scorer.use_sequence=False \
        sampling.batch_size=1 \
        sampling.number_of_batches=10


ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.single_aa_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/single_aa_score_w_seq" \
        scorer.use_sequence=True \
        sampling.batch_size=1 \
        sampling.number_of_batches=10


ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.single_aa_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/single_aa_score_wo_seq" \
        scorer.use_sequence=False \
        sampling.batch_size=1 \
        sampling.number_of_batches=10


# score from pdb
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/autoregressive_score_w_seq_pdb_bz10" \
        scorer.use_sequence=True \
        sampling.batch_size=10 \
        sampling.number_of_batches=1 > autoregressive_score_w_seq_pdb_bz10.log

# score from given sequence
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/autoregressive_score_w_seq_fasta_bz10" \
        scorer.use_sequence=True \
        sampling.batch_size=10 \
        sampling.number_of_batches=1 \
        scorer.customized_seq='GMSSISLPEFLLELLSDPKYEDYIKWVSDNGEFELKNPEAVAKLWGEKKGLPDMNYEKMYKELKKYEKKKIIEKVKGKPNVYKFVNYPEILNP' > autoregressive_score_w_seq_fasta_bz10.log

diff autoregressive_score_w_seq_pdb_bz10.log autoregressive_score_w_seq_fasta_bz10.log || echo Never mind