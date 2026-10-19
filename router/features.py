"""Future syndrome and experiment feature extraction.

Planned features include event count/density, activity by round, number of
active rounds, maximum round activity, distance, rounds, and basis. Use
round_widths to mask padded positions before counting or computing density.
Never use label, decoder predictions, or derived correctness as inputs.
"""
