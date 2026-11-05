"""Week 2 Tasks 1-2: leakage-safe feature extraction (scaffold only).

Implement reusable extraction here before filling notebook 01. Read each
flattened syndrome with time_steps and width; count events only within each
round_widths[t] real-position slice. Define event count, density, per-time-step
activity, active time steps, and maximum activity. Add distance, rounds, and
basis with stable named columns; document a fixed-length activity representation.

Build X from an explicit source-field allowlist: syndrome, round_widths,
time_steps, width, distance, rounds, and basis. The output must never contain
label, pymatching, correlated_matching, decoder correctness, y_router, or
experiment. Changing only those forbidden fields must leave X unchanged.
See configs/features.yaml and
docs/WEEK_2_TASK_GUIDE.md. No feature extractor is implemented yet.
"""
