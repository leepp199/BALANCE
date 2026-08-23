# Final paper-facing module names

The implementation aliases are retained for reproducibility, but the manuscript uses:

* **LSRB — Latent Structure Reference Bank**: a dataset-level bank of recurring
  references in the task-adapted latent space. It is not a time/frequency bank.
* **CANA — Capacity-Aware Novel-Class Assignment**: the open-world assignment
  operation that allocates rejected samples to emerging-class prototypes while
  respecting the estimated stream capacity.

Legacy aliases retained in code and raw logs: `DFSB = LSRB` and
`balanced_kmeans/BCD = CANA`. The mapping is stated once in the reproducibility
paragraph; LSRB and CANA are used everywhere else in the manuscript.
