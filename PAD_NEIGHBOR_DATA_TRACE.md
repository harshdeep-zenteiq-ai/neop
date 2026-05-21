# pad_neighbor_data() Inner Working - Step-by-Step Trace

## Call
```python
pad_neighbor_data([
    data_module_jax.train_data.data_list,  # 5 samples
    data_module_jax.test_data.data_list,   # 2 samples
])
```

---

## Line 123: Flatten Data Lists
```python
all_samples = [s for dl in data_lists for s in dl]
# PRINT: all_samples has 7 samples (5 train + 2 test)
# all_samples = [sample0, sample1, sample2, sample3, sample4, test0, test1]
```

---

## Lines 125-126: Find Maximum Edge Counts
```python
max_in = max(s['neighbors_in']['neighbors_index'].shape[0] for s in all_samples)
# PRINT: Scans all 7 samples' neighbors_in sizes
# PRINT: max_in = 450  (e.g., sample3 has most input edges)

max_out = max(s['neighbors_out']['neighbors_index'].shape[0] for s in all_samples)
# PRINT: Scans all 7 samples' neighbors_out sizes
# PRINT: max_out = 520  (e.g., test1 has most output edges)
```

---

## Line 129: Loop Through All Samples
```python
for s in all_samples:  # 7 iterations (all samples)
    # PRINT: Processing sample 0
    # PRINT: Processing sample 1
    # ... etc
```

---

## Line 130: Loop Through Neighbor Types
```python
for key, max_edges in [('neighbors_in', 450), ('neighbors_out', 520)]:
    # PRINT Iteration 1: key='neighbors_in', max_edges=450
    # PRINT Iteration 2: key='neighbors_out', max_edges=520
```

---

## Lines 131-135: Extract & Analyze Neighbor Data
```python
nbrs = s[key]
# PRINT: nbrs = {'neighbors_index': [...], 'neighbors_row_splits': [...]}

idx = np.asarray(nbrs['neighbors_index'])
# PRINT: idx.shape = (340,)  [e.g., 340 actual edges in this sample's neighbors_in]

splits = np.asarray(nbrs['neighbors_row_splits'])
# PRINT: splits.shape = (65,)  [e.g., 64 output nodes + 1 = 65 split points]

n_real = idx.shape[0]
# PRINT: n_real = 340  [actual edges count]

n_out = splits.shape[0] - 1
# PRINT: n_out = 64  [number of output nodes]
```

---

## Lines 137-142: Pad Data
```python
counts = splits[1:] - splits[:-1]
# PRINT: counts = [5, 3, 8, 7, ...] shape=(64,)  [edges-per-node]

seg_ids = np.repeat(np.arange(n_out), counts)
# PRINT: seg_ids = [0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3, ...]
# PRINT: seg_ids.shape = (340,)  [one segment ID per edge]

pad_len = max_edges - n_real
# PRINT: pad_len = 450 - 340 = 110  [padding needed]

idx_padded = np.concatenate([idx, np.zeros(pad_len, dtype=idx.dtype)])
# PRINT: idx_padded.shape = (450,)  [original 340 + 110 zeros as padding]

seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, dtype=seg_ids.dtype)])
# PRINT: seg_ids_padded.shape = (450,)  [original 340 seg_ids + 110 copies of n_out=64]
# PRINT: (Dummy segment 64 is safe; only segment 0-63 have real data)
```

---

## Lines 151-155: Store Transformed Data
```python
s[key] = {
    'neighbors_index': idx_padded,      # [450] numpy array
    'segment_ids': seg_ids_padded,      # [450] numpy array
    'counts': counts,                   # [64] numpy array
}
# PRINT: Replaced s['neighbors_in'] with padded version
# PRINT: Original 'neighbors_row_splits' DROPPED (no longer needed)
```

---

## Final State
After all 7 samples × 2 neighbor types processed:

```python
# Each sample's structure changed from:
sample['neighbors_in'] = {
    'neighbors_index': [n_real],          ✗ Variable length (e.g., 340)
    'neighbors_row_splits': [n_out+1],    ✗ Variable, dropped
}

# To:
sample['neighbors_in'] = {
    'neighbors_index': [max_in=450],      ✓ Fixed length, padded
    'segment_ids': [max_in=450],          ✓ Fixed length, padded
    'counts': [n_out=64],                 ✓ Edge count per node
}

# Same for 'neighbors_out' with max_out=520
```

---

## Return
```python
return max_in, max_out
# PRINT: Returning (450, 520)
# These values are max edge counts across ALL 7 samples, train + test
```

---

## Why This Works
- **JIT Compatibility:** Fixed shapes allow `@jax.jit` to compile
- **Segment Aggregation:** `segment_ids` enables mean-pooling via `segment_csr`
- **Dummy Segments:** Padding with segment ID `n_out=64` is discarded in reduction
- **Shared Limits:** Train and test use same `max_in/max_out` ⟹ consistent compilation
