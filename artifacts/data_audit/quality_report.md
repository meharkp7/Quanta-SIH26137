# Training-data quality report - corpus_delhi

Sampling: manifest metrics = ALL 250 episodes; episode-deep metrics = stratified sample of 12 episodes (['ep-0025', 'ep-0049', 'ep-0077', 'ep-0053', 'ep-0117', 'ep-0133', 'ep-0182', 'ep-0176', 'ep-0170', 'ep-0231', 'ep-0223', 'ep-0238']); windows built for ['ep-0025', 'ep-0182', 'ep-0231'].

## Headline coverage (manifest, all 250 episodes)
| split | eps | speed mean | speed min-max | trav mean | trav min-max |
|---|---|---|---|---|---|
| train | 150 | 0.0347 | 0.0107-0.0691 | 0.0311 | 0.0088-0.0608 |
| validation | 50 | 0.0566 | 0.0246-0.0684 | 0.0490 | 0.0203-0.0587 |
| test | 50 | 0.0512 | 0.0263-0.0662 | 0.0452 | 0.0219-0.0590 |

## Sampled per-horizon x kind coverage (labels.jsonl stream)
| horizon/kind | valid | total | coverage |
|---|---|---|---|
| 300s/realized_traversal | 11765 | 331632 | 0.0355 |
| 300s/speed_proxy | 12600 | 331632 | 0.0380 |
| 600s/realized_traversal | 10416 | 331632 | 0.0314 |
| 600s/speed_proxy | 11804 | 331632 | 0.0356 |
| 900s/realized_traversal | 8467 | 331632 | 0.0255 |
| 900s/speed_proxy | 10620 | 331632 | 0.0320 |

## Ranked data-side limits (severity + evidence)
1. **[HIGH] Extreme supervision sparsity** - train speed coverage mean 0.0347, traversal 0.0311; sampled obs missing 0.954; only 9 issue-times/episode
2. **[HIGH] 9-windows/episode thinness** - 27 windows over 3 sampled episodes; ~2250 windows corpus-wide dilute incident supervision
3. **[MEDIUM-HIGH] Train/val/test density drift** - val/test mean speed coverage 0.0566/0.0512 vs train 0.0347 (1.63x)
4. **[MEDIUM-HIGH] Traversal-target heavy tail + scale mismatch** - sampled trav p50 5.0s, p99 41.7s, max 570.0s vs speed_ratio~[0,1]
5. **[MEDIUM] Incident supervision thin** - incident-window fraction of valid labels 0.0000 (sampled)
6. **[MEDIUM] Per-map graph-size spread + single-map val/test** - edges [2082, 4778, 1929, 5465, 2137]; val/test rely on one map each
7. **[LOW-MEDIUM] No scaler clipping on skewed features** - FeatureScaler is pure z-score; halting/occupancy skew reported per-feature above

## Leakage / causality
- episode overlap t/v/v/t: 0/0/0; fingerprints unique: True; scenarios single-split: True
- causal asserts over 27 windows: 0 failures; avail<target violations: 0

## Scaler / outliers
- speed_ratio: mean=0.8076 std=0.1984 skew=-1.31 |z|>5 frac=0.0000 (no clipping in FeatureScaler)
- occupancy: mean=0.001135 std=0.001122 skew=2.57 |z|>5 frac=0.0045 (no clipping in FeatureScaler)
- halting: mean=0.01172 std=0.1029 skew=8.99 |z|>5 frac=0.0122 (no clipping in FeatureScaler)
- observation_age_min: mean=11.94 std=6.017 skew=-0.13 |z|>5 frac=0.0000 (no clipping in FeatureScaler)
- missing: mean=0.9636 std=0.1872 skew=-4.95 |z|>5 frac=0.0364 (no clipping in FeatureScaler)
- known_closed: mean=0 std=1 skew=0.00 |z|>5 frac=0.0000 (no clipping in FeatureScaler)

## Sampler proposal (not applied)
- incident fraction of valid labels: 0.0000
- route ~0.15 of each batch from incident-affected (edge,bucket) pairs OR scale incident-window loss by ~1000.0x capped at 8x; measured incident fraction=0.0000

## Incident spillover (post-hoc, exact numbers)

Valid labels whose target falls inside any event effect window (any edge), per sampled event episode: ep-0077=0.6450 ep-0053=0.6897 ep-0117=0.0175 ep-0133=0.0048 ep-0176=0.0000 ep-0231=0.1076 ep-0223=0.1171 ep-0238=0.5756.
Exact (disrupted-edge, effect-bucket) valid labels: ~0 in all 12 sampled episodes (closed edges emit missing=1 rows; incidents touch <=9 edges).
