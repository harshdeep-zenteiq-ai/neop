# pad_neighbor_data() - Actual Execution Output with Print Statements

## Function Call
```python
pad_neighbor_data([
    data_module_jax.train_data.data_list,  # 5 training samples
    data_module_jax.test_data.data_list,   # 2 test samples
])
```

---

## Execution Trace with Prints After Every Line

### Line 123: Flatten Data Lists
```
>>> Line 123: all_samples = [s for dl in data_lists for s in dl]
    RESULT: Flattened 2 lists into all_samples with 7 total samples
```

### Lines 125-126: Find Maximum Edge Counts
```
>>> Line 125: max_in = max(s['neighbors_in']['neighbors_index'].shape[0] for s in all_samples)
    RESULT: max_in = 16681

>>> Line 126: max_out = max(s['neighbors_out']['neighbors_index'].shape[0] for s in all_samples)
    RESULT: max_out = 16681
```

---

## Sample 0 (Train Sample 0)

### neighbors_in Processing
```
>>> Line 131: nbrs = s[key]
    OK: nbrs keys: ['neighbors_index', 'neighbors_row_splits']

>>> Line 132: idx = np.asarray(nbrs['neighbors_index'])
    Shape: (16681,), dtype: int32, first 5: [2691 1132 2697 1922 2694]

>>> Line 133: splits = np.asarray(nbrs['neighbors_row_splits'])
    Shape: (32769,), dtype: int32, first 5: [0 0 0 0 0]

>>> Line 134: n_real = idx.shape[0]
    n_real = 16681

>>> Line 135: n_out = splits.shape[0] - 1
    n_out = 32768

>>> Line 137: counts = splits[1:] - splits[:-1]
    Shape: (32768,), min: 0, max: 8, sum: 16681

>>> Line 138: seg_ids = np.repeat(np.arange(n_out), counts)
    Shape: (16681,), first 5: [36 37 37 38 38]

>>> Line 139: pad_len = max_edges - n_real
    pad_len = 0

>>> Line 141: idx_padded = np.concatenate([idx, np.zeros(pad_len, ...)])
    Shape: (16681,), Last 5 values: [3349 2460 3348 1734 2586]

>>> Line 142: seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, ...)])
    Shape: (16681,), Last 5 values: [32244 32245 32245 32246 32247]

>>> Line 151-155: s['neighbors_in'] = {updated}
    OK: Updated s['neighbors_in'] with 3 keys
```

### neighbors_out Processing
```
>>> Line 131: nbrs = s[key]
    OK: nbrs keys: ['neighbors_index', 'neighbors_row_splits']

>>> Line 132: idx = np.asarray(nbrs['neighbors_index'])
    Shape: (16681,), dtype: int32, first 5: [3106 3107 3139 4130 4131]

>>> Line 133: splits = np.asarray(nbrs['neighbors_row_splits'])
    Shape: (3587,), dtype: int32, first 5: [ 0  5  9 13 17]

>>> Line 134: n_real = idx.shape[0]
    n_real = 16681

>>> Line 135: n_out = splits.shape[0] - 1
    n_out = 3586

>>> Line 137: counts = splits[1:] - splits[:-1]
    Shape: (3586,), min: 2, max: 8, sum: 16681

>>> Line 138: seg_ids = np.repeat(np.arange(n_out), counts)
    Shape: (16681,), first 5: [0 0 0 0 0]

>>> Line 139: pad_len = max_edges - n_real
    pad_len = 0

>>> Line 141: idx_padded = np.concatenate([idx, np.zeros(pad_len, ...)])
    Shape: (16681,), Last 5 values: [12797 13821 14844 14845 14877]

>>> Line 142: seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, ...)])
    Shape: (16681,), Last 5 values: [3584 3585 3585 3585 3585]

>>> Line 151-155: s['neighbors_out'] = {updated}
    OK: Updated s['neighbors_out'] with 3 keys
```

---

## Sample 1 (Train Sample 1)

### neighbors_in Processing
```
>>> Line 131: nbrs = s[key]
    OK: nbrs keys: ['neighbors_index', 'neighbors_row_splits']

>>> Line 132: idx = np.asarray(nbrs['neighbors_index'])
    Shape: (16007,), dtype: int32, first 5: [   0 1798 1132 2691 1922]

>>> Line 133: splits = np.asarray(nbrs['neighbors_row_splits'])
    Shape: (32769,), dtype: int32, first 5: [0 0 0 0 0]

>>> Line 134: n_real = idx.shape[0]
    n_real = 16007

>>> Line 135: n_out = splits.shape[0] - 1
    n_out = 32768

>>> Line 137: counts = splits[1:] - splits[:-1]
    Shape: (32768,), min: 0, max: 9, sum: 16007

>>> Line 138: seg_ids = np.repeat(np.arange(n_out), counts)
    Shape: (16007,), first 5: [36 69 70 70 71]

>>> Line 139: pad_len = max_edges - n_real
    pad_len = 674  ← PADDING NEEDED (sample 0 had max at 16681)

>>> Line 141: idx_padded = np.concatenate([idx, np.zeros(pad_len, ...)])
    Shape: (16681,), Last 5 values: [0 0 0 0 0]  ← Padded with zeros

>>> Line 142: seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, ...)])
    Shape: (16681,), Last 5 values: [32768 32768 32768 32768 32768]  ← Dummy segment ID

>>> Line 151-155: s['neighbors_in'] = {updated}
    OK: Updated s['neighbors_in'] with 3 keys
```

### neighbors_out Processing
```
>>> Line 131: nbrs = s[key]
    OK: nbrs keys: ['neighbors_index', 'neighbors_row_splits']

>>> Line 132: idx = np.asarray(nbrs['neighbors_index'])
    Shape: (16007,), dtype: int32, first 5: [  36 1060 1061 1092 4252]

>>> Line 133: splits = np.asarray(nbrs['neighbors_row_splits'])
    Shape: (3587,), dtype: int32, first 5: [ 0  4  8 12 17]

>>> Line 134: n_real = idx.shape[0]
    n_real = 16007

>>> Line 135: n_out = splits.shape[0] - 1
    n_out = 3586

>>> Line 137: counts = splits[1:] - splits[:-1]
    Shape: (3586,), min: 2, max: 8, sum: 16007

>>> Line 138: seg_ids = np.repeat(np.arange(n_out), counts)
    Shape: (16007,), first 5: [0 0 0 0 1]

>>> Line 139: pad_len = max_edges - n_real
    pad_len = 674

>>> Line 141: idx_padded = np.concatenate([idx, np.zeros(pad_len, ...)])
    Shape: (16681,), Last 5 values: [0 0 0 0 0]

>>> Line 142: seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, ...)])
    Shape: (16681,), Last 5 values: [3586 3586 3586 3586 3586]

>>> Line 151-155: s['neighbors_out'] = {updated}
    OK: Updated s['neighbors_out'] with 3 keys
```

---

## Sample 2 (Train Sample 2)

### neighbors_in Processing
```
>>> Line 132: idx Shape: (16111,), first 5: [1922 2694  964 2694 2718]
>>> Line 133: splits Shape: (32769,)
>>> Line 134: n_real = 16111
>>> Line 135: n_out = 32768
>>> Line 137: counts Shape: (32768,), min: 0, max: 11, sum: 16111
>>> Line 138: seg_ids Shape: (16111,), first 5: [39 39 40 40 40]
>>> Line 139: pad_len = 570  ← 16681 - 16111
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [32768 32768 32768 32768 32768]
```

### neighbors_out Processing
```
>>> Line 132: idx Shape: (16111,), first 5: [2084 3108 3109 3140 1082]
>>> Line 133: splits Shape: (3587,)
>>> Line 139: pad_len = 570
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [3586 3586 3586 3586 3586]
```

---

## Sample 3 (Train Sample 3)

### neighbors_in Processing
```
>>> Line 132: idx Shape: (15832,), first 5: [1855  758 1854  256 1855]
>>> Line 139: pad_len = 849  ← Largest padding requirement
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [32768 32768 32768 32768 32768]
```

### neighbors_out Processing
```
>>> Line 132: idx Shape: (15832,)
>>> Line 139: pad_len = 849
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [3586 3586 3586 3586 3586]
```

---

## Sample 4 (Train Sample 4)

### neighbors_in Processing
```
>>> Line 132: idx Shape: (16330,), first 5: [1218 1218 1218 1218 2709]
>>> Line 139: pad_len = 351
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [32768 32768 32768 32768 32768]
```

### neighbors_out Processing
```
>>> Line 132: idx Shape: (16330,)
>>> Line 139: pad_len = 351
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [3586 3586 3586 3586 3586]
```

---

## Sample 5 (Test Sample 0)

### neighbors_in Processing
```
>>> Line 132: idx Shape: (16457,), first 5: [2709 1842 2706  924 3010]
>>> Line 139: pad_len = 224
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [32768 32768 32768 32768 32768]
```

### neighbors_out Processing
```
>>> Line 132: idx Shape: (16457,)
>>> Line 139: pad_len = 224
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [3586 3586 3586 3586 3586]
```

---

## Sample 6 (Test Sample 1)

### neighbors_in Processing
```
>>> Line 132: idx Shape: (16101,), first 5: [1922 2697 1922 2694  964]
>>> Line 139: pad_len = 580
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [32768 32768 32768 32768 32768]
```

### neighbors_out Processing
```
>>> Line 132: idx Shape: (16101,)
>>> Line 139: pad_len = 580
>>> Line 141: idx_padded Shape: (16681,), Last 5: [0 0 0 0 0]
>>> Line 142: seg_ids_padded Shape: (16681,), Last 5: [3586 3586 3586 3586 3586]
```

---

## Final Return Statement

```
>>> Line 158: return max_in, max_out
    Returning: (16681, 16681)
```

---

## Summary

| Metric | Value |
|--------|-------|
| Total Samples Processed | 7 (5 train + 2 test) |
| max_in_edges | 16681 |
| max_out_edges | 16681 |
| Largest Padding Needed (neighbors_in) | 849 edges (Sample 3) |
| Smallest Padding (neighbors_in) | 0 edges (Sample 0) |
| Average neighbors_in edges | 16,219 |
| Average neighbors_out edges | 16,219 |

## Key Observations

1. **Uniform max_edges:** Both neighbors_in and neighbors_out share max_edges=16681, enabling single JIT compilation
2. **Variable padding:** Different samples require different amounts of padding (0-849 edges)
3. **Dummy segments:** Padding dummy indices (0) with segment IDs set to n_out (32768 or 3586) are safely ignored during aggregation
4. **Structure transformation:** Each sample's neighbor dict changed from 2 keys (neighbors_index, neighbors_row_splits) to 3 keys (neighbors_index, segment_ids, counts)
5. **No mutation after padding:** Original neighbors_row_splits dropped (no longer needed at runtime)
