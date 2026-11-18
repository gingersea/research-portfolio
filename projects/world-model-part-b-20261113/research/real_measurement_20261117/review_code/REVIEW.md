{
  "fixture_calculation": {
    "test_unlabeled_zero_is_not_an_occluded_negative": {
      "positive": 1,
      "negative": 1,
      "accepted": 2,
      "correct": 1,
      "false_visible": 1,
      "returned_points": 1,
      "recovered": 1,
      "strict_recovered": 0
    }
  },
  "findings": [
    {
      "file": "research/real_measurement_20261117/analyze.py",
      "line": 34,
      "function": "main",
      "evidence": "assert not (prediction['confidence_accepted'] & ~prediction['native']).any()",
      "finding": "Confidence gate incorrectly filters native predictions, violating reverse reciprocity as an acceptance gate, not coordinate repair."
    },
    {
      "file": "research/real_measurement_20261117/run_models.py",
      "line": 102,
      "function": "main",
      "evidence": "reverse_query=np.array([0,candidate[0],candidate[1]],np.float32)",
      "finding": "Reverse query uses (0, x, y) format but expects (t, y, x) in reciprocal_accept, causing coordinate scaling mismatch."
    },
    {
      "file": "research/real_measurement_20261117/prepare.py",
      "line": 68,
      "function": "main",
      "evidence": "xy=ann['trajs_2d'].astype(np.float32)*np.array([512/width,288/height],np.float32)",
      "finding": "Coordinate scaling applies 512/width and 288/height but reciprocal_accept uses 256/width and 256/height, causing normalization mismatch."
    }
  ],
  "limits": [
    "No concrete defect found in sparse.py fixture calculation",
    "Test_unlabeled_zero_is_not_an_occluded_negative passes with correct denominator",
    "Reverse reciprocity gate in run_models.py uses incorrect coordinate format",
    "Coordinate scaling in prepare.py and reciprocal_accept uses different normalization factors",
    "Confidence gate in analyze.py incorrectly filters native predictions"
  ]
}