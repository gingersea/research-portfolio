# Synthetic stress results

| Scene patch (px) | Input | Method | Mean IoU [95% CI] | Mean centroid error (px) | Nonempty fraction |
|---:|---|---|---|---:|---:|
| 24 | native | learned | 0.1616 [0.1340, 0.1889] | 120.9299 | 0.4570 |
| 24 | native | color | 0.6654 [0.6470, 0.6826] | 1.2017 | 1.0000 |
| 24 | warp96 | learned | 0.0000 [0.0000, 0.0000] | 222.2791 | 0.0000 |
| 24 | warp96 | color | 0.6589 [0.6434, 0.6743] | 0.6455 | 1.0000 |
| 24 | letterbox96 | learned | 0.0000 [0.0000, 0.0000] | 222.2791 | 0.0000 |
| 24 | letterbox96 | color | 0.5927 [0.5757, 0.6084] | 0.8167 | 1.0000 |
| 48 | native | learned | 0.7954 [0.7728, 0.8171] | 0.3766 | 1.0000 |
| 48 | native | color | 0.8028 [0.7835, 0.8214] | 2.0642 | 1.0000 |
| 48 | warp96 | learned | 0.2986 [0.2629, 0.3349] | 80.9271 | 0.6367 |
| 48 | warp96 | color | 0.7601 [0.7428, 0.7763] | 1.5744 | 1.0000 |
| 48 | letterbox96 | learned | 0.1606 [0.1334, 0.1875] | 120.8957 | 0.4570 |
| 48 | letterbox96 | color | 0.7319 [0.7152, 0.7481] | 1.7395 | 1.0000 |
| 96 | native | learned | 0.9650 [0.9615, 0.9682] | 0.2242 | 1.0000 |
| 96 | native | color | 0.8932 [0.8745, 0.9106] | 3.1234 | 1.0000 |
| 96 | warp96 | learned | 0.8339 [0.8207, 0.8463] | 0.3079 | 1.0000 |
| 96 | warp96 | color | 0.8065 [0.7873, 0.8248] | 3.5659 | 1.0000 |
| 96 | letterbox96 | learned | 0.7963 [0.7744, 0.8176] | 0.3267 | 1.0000 |
| 96 | letterbox96 | color | 0.8079 [0.7884, 0.8262] | 3.5963 | 1.0000 |

## Paired IoU contrasts

| Scene patch | Method | Contrast | Mean difference [95% CI] |
|---:|---|---|---|
| 24 | learned | warp96 minus native | -0.1616 [-0.1889, -0.1340] |
| 24 | learned | letterbox96 minus native | -0.1616 [-0.1889, -0.1340] |
| 24 | color | warp96 minus native | -0.0066 [-0.0247, 0.0118] |
| 24 | color | letterbox96 minus native | -0.0728 [-0.0892, -0.0557] |
| 48 | learned | warp96 minus native | -0.4968 [-0.5259, -0.4683] |
| 48 | learned | letterbox96 minus native | -0.6348 [-0.6611, -0.6093] |
| 48 | color | warp96 minus native | -0.0427 [-0.0521, -0.0330] |
| 48 | color | letterbox96 minus native | -0.0708 [-0.0803, -0.0612] |
| 96 | learned | warp96 minus native | -0.1311 [-0.1427, -0.1197] |
| 96 | learned | letterbox96 minus native | -0.1687 [-0.1885, -0.1493] |
| 96 | color | warp96 minus native | -0.0867 [-0.0927, -0.0800] |
| 96 | color | letterbox96 minus native | -0.0853 [-0.0912, -0.0788] |
