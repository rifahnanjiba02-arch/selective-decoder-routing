"""Week 2 Task 3 classical-router test scaffold; no tests are implemented yet.

When implementing src/router/models/classical.py, add small synthetic tests for:
- the same feature columns and rescue target used by every candidate;
- train-only fitting of models, encoding, scaling, and imputation;
- both target classes present in the tiny training example;
- one finite probability in [0, 1] per validation row;
- explicit preprocessing columns with remainder='drop' and no passthrough;
- injected forbidden columns rejected or dropped before model fitting;
- validation-only category/extreme value does not alter train-fitted state.

See docs/WEEK_2_TASK_GUIDE.md. This file intentionally has no passing placeholder
tests: a green test run alone must not be mistaken for Week 2 model coverage.
"""
