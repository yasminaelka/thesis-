# SCM-based fairness and faithful explainability for legal document classification

Code for the MSc Information Studies (Data Science) thesis *SCM-based fairness
and faithful explainability for legal document classification* (University of
Amsterdam, 2026), evaluated on the ECtHR alleged-violations corpus
(`coastalcph/lex_glue`, configuration `ecthr_a`).

## Summary

The central result is a dissociation. Across five seeds, SCM contrastive
regularisation of LegalBERT leaves demographic fairness (DPD and EOD) unchanged
and does not reliably change macro F1, but consistently degrades SHAP
explanation sufficiency. A shuffled-pair control reproduces all three effects,
so the pattern follows from contrastive regularisation of any kind rather than
from the warmth and competence structure of the Stereotype Content Model.
Because one property stays fixed while the other moves, explanation faithfulness
could not have served as a proxy for fairness in this setting.

All experiments ran on the SURF Snellius cluster (partition `gpu_a100`, one
A100 per job).

## Repository structure

```
training/
  run_contrastive_v2.py            Main pipeline: trains baseline, SCM, and
                                   shuffled-pair conditions; tunes per-article
                                   thresholds; computes SHAP and IG
                                   faithfulness. Supports --encoder
                                   {legal-bert, bert, roberta} (Sections 3.3-3.6;
                                   cross-encoder runs in Appendix F).
  run_contrastive_wordpair.py      Word-pair debiasing control (Appendix G).
                                   Same contrastive loss as the main script with
                                   the extra --pairs {wordpair_gender,
                                   wordpair_eth} conditions.
  run_instance_weighted_scm.py     Instance-weighted variant (Appendix E):
                                   documents the per-document w_i weighting on
                                   top of the main pipeline (see note below).

fairness/
  fairness_ci_eod_multiseed.py     Per-seed DPD and EOD over the seven reliable
                                   articles, with 95% bootstrap CIs
                                   (Section 4, Appendix B). Authoritative
                                   grouping: whole-word regex, gender 931/69,
                                   ethnicity 136/864.
  fairness_ci_eod_ner.py           NER-based ethnicity robustness check
                                   (protected 59/941; Section 4, Appendix B).
  ner_groups.py                    Builds the spaCy NER ethnicity grouping
                                   consumed by fairness_ci_eod_ner.py.
  check_group_sizes_v2.py          Group-size analysis justifying the keyword
                                   choices (whole-word matching, targeted
                                   ethnicity set, exclusion of nationality
                                   adjectives; Section 3.6).
  recompute_iw_fairness_corrected.py
                                   Recomputes the Appendix E instance-weighted
                                   DPD under the corrected grouping from the
                                   saved lambda=0.5 checkpoint (see note below).

faithfulness/
  run_ig_multiseed.py              Integrated Gradients faithfulness, the
                                   cross-method check on the SHAP sufficiency
                                   result (Table 4).
  aggregate_ig_multiseed.py        Aggregates the IG runs across seeds.

lambda_sweeps/
  lambda_sweep_fairness.py         Fairness across regularisation strength lambda
                                   (Table 5).
  lambda_sweep_faithfulness.py     Faithfulness across lambda (Appendix A,
                                   Figure 6).

aggregation/
  aggregate_multiseed.py           Aggregates baseline, SCM, and shuffled
                                   results across the five seeds (Tables 2-4).
  wilcoxon_job1.py                 Per-instance Wilcoxon test on the faithfulness
                                   arrays.

figures/
  make_dissociation_fig.py         Figure 5 (the dissociation across conditions).
  make_fig_lambda.py               Faithfulness-lambda figure.

slurm/
  submit_5seeds.sh                 Launches the five-seed SCM and shuffled runs.
  run_wordpair.slurm               Launches the word-pair control.
  submit_eod.sh                    Launches the fairness/EOD evaluation.
  run_ig_multiseed.slurm           Launches the IG runs.
  submit_lam_sweep.sh              Launches the fairness lambda sweep.
  submit_lam_faith.sh              Launches the faithfulness lambda sweep.
```

## Reproducing the results

Seeds are 42, 13, 7, 21, 100. Conditions are baseline, SCM, shuffled-pair, and
word-pair. The order below mirrors the experimental procedure in Section 3.7.

1. **Train and evaluate the main conditions** (baseline, SCM, shuffled), five
   seeds each. This also produces the SHAP faithfulness numbers:
   ```
   sbatch slurm/submit_5seeds.sh
   ```
2. **Fairness with bootstrap CIs**, per seed:
   ```
   python fairness/fairness_ci_eod_multiseed.py --pairs scm --seed 42
   ```
   and the NER robustness check:
   ```
   python fairness/fairness_ci_eod_ner.py --pairs scm --seed 42 --ethnicity_source ner
   ```
3. **Integrated Gradients** cross-method check:
   ```
   sbatch slurm/run_ig_multiseed.slurm
   ```
4. **Lambda sweeps**:
   ```
   sbatch slurm/submit_lam_sweep.sh
   sbatch slurm/submit_lam_faith.sh
   ```
5. **Word-pair control** (Appendix G):
   ```
   sbatch slurm/run_wordpair.slurm
   ```
6. **Aggregate and plot**:
   ```
   python aggregation/aggregate_multiseed.py
   python figures/make_dissociation_fig.py
   python figures/make_fig_lambda.py
   ```

The scripts use absolute Snellius paths (output under
`/gpfs/home6/yelkacemi/output`) and the SLURM scripts assume the Python files
sit in the working directory, as they did when the experiments ran. The
repository is organised into folders for readability; to re-execute, adjust the
paths and the script locations for your environment.

## Notes on two analyses

**Word-pair control faithfulness.** `run_contrastive_wordpair.py` writes
faithfulness output alongside its DPD output, but only the DPD and F1 numbers
are used in the thesis (Appendix G). The main faithfulness results come from
`run_contrastive_v2.py`, which uses a real-token masking procedure for
sufficiency and comprehensiveness. The two scripts differ in that masking
detail, so the word-pair faithfulness output should not be compared directly to
Table 4.

**Instance weighting (Appendix E).** `run_instance_weighted_scm.py` documents
the per-document w_i weighting as a modification of the main pipeline rather
than duplicating the full training code; the new logic is fenced with
`INSTANCE WEIGHTING: NEW CODE` markers. The Appendix E fairness numbers were
recomputed from the saved lambda=0.5 checkpoint under the corrected keyword
grouping using `recompute_iw_fairness_corrected.py`, which reuses the exact
grouping and metric functions from `fairness_ci_eod_multiseed.py`.

## Environment

Python 3.10. Key packages:

```
torch          2.12.0
transformers   5.9.0
datasets       4.8.5
numpy          2.2.6
scikit-learn   1.7.2
shap           0.49.1
```

Models: LegalBERT (`nlpaueb/legal-bert-base-uncased`), `bert-base-uncased`,
`roberta-base`. SHAP uses KernelSHAP; faithfulness is also computed with
Integrated Gradients.

Note on data loading: with `datasets` 4.x the `trust_remote_code` argument is
ignored, and the LexGLUE `ecthr_a` configuration loads directly. Each document
is a list of fact paragraphs joined into a single string, then truncated with a
head and tail strategy (first 256 and last 256 subword tokens).
