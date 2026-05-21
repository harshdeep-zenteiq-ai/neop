# pad_neighbor_data() - Minimal Dummy Example Execution

## Setup
- **Vertices:** 5
- **Grid Points:** 8 (2×2×2 query resolution)
- **Training Samples:** 5
- **Testing Samples:** 2
- **Total Samples:** 7

### Data Structure
```
neighbors_in:
  - neighbors_index: indices into 5 vertices (values 0-4)
  - neighbors_row_splits: split points for 9 query points
  - Edges per sample: 6

neighbors_out:
  - neighbors_index: indices into 8 query points (values 0-7)
  - neighbors_row_splits: split points for 5 vertices
  - Edges per sample: 13
```

---

## LINE 123: Flatten Data Lists

```
>>> LINE 123: all_samples = [s for dl in data_lists for s in dl]
    OUTPUT: all_samples = [sample_0, sample_1, ..., sample_6]
    Total samples flattened: 7
```

---

## LINE 125: Find max_in

```
>>> LINE 125: max_in = max(s['neighbors_in']['neighbors_index'].shape[0] for s in all_samples)
    Scanning all 7 samples...
      Sample 0: 6 neighbors_in edges
      Sample 1: 6 neighbors_in edges
      Sample 2: 6 neighbors_in edges
      Sample 3: 6 neighbors_in edges
      Sample 4: 6 neighbors_in edges
      Sample 5: 6 neighbors_in edges
      Sample 6: 6 neighbors_in edges
    OUTPUT: max_in = 6
```

---

## LINE 126: Find max_out

```
>>> LINE 126: max_out = max(s['neighbors_out']['neighbors_index'].shape[0] for s in all_samples)
    Scanning all 7 samples...
      Sample 0: 13 neighbors_out edges
      Sample 1: 13 neighbors_out edges
      Sample 2: 13 neighbors_out edges
      Sample 3: 13 neighbors_out edges
      Sample 4: 13 neighbors_out edges
      Sample 5: 13 neighbors_out edges
      Sample 6: 13 neighbors_out edges
    OUTPUT: max_out = 13
```

---

## SAMPLE 0 - neighbors_in Processing

```
>>> LINE 131: nbrs = s[key]
    OUTPUT: nbrs dict with 2 keys: ['neighbors_index', 'neighbors_row_splits']

>>> LINE 132: idx = np.asarray(nbrs['neighbors_index'])
    OUTPUT: idx = [1, 3, 0, 1, 4, 0]
            shape = (6,), dtype = int32

>>> LINE 133: splits = np.asarray(nbrs['neighbors_row_splits'])
    OUTPUT: splits = [0, 0, 1, 1, 2, 3, 4, 5, 5, 6]
            shape = (10,), dtype = int32

>>> LINE 134: n_real = idx.shape[0]
    OUTPUT: n_real = 6 (actual edges in this sample)

>>> LINE 135: n_out = splits.shape[0] - 1
    OUTPUT: n_out = 9 (number of output nodes)

>>> LINE 137: counts = splits[1:] - splits[:-1]
    OUTPUT: counts = [0, 1, 0, 1, 1, 1, 1, 0, 1]
            (edges per output node)

>>> LINE 138: seg_ids = np.repeat(np.arange(n_out), counts)
    OUTPUT: seg_ids = [1, 3, 4, 5, 6, 8]
            (segment ID per edge, maps edges to nodes)

>>> LINE 139: pad_len = max_edges - n_real
    OUTPUT: pad_len = 6 - 6 = 0
            (No padding needed - this sample has max edges)

>>> LINE 141: idx_padded = np.concatenate([idx, np.zeros(pad_len, ...)])
    OUTPUT: idx_padded = [1, 3, 0, 1, 4, 0]
            shape = (6,)

>>> LINE 142: seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, ...)])
    OUTPUT: seg_ids_padded = [1, 3, 4, 5, 6, 8]
            shape = (6,)
            (last 0 values are dummy segment 9)

>>> LINE 151-155: s['neighbors_in'] = { updated_dict }
    OUTPUT: Updated s['neighbors_in'] structure:
      OLD:  {'neighbors_index': 6 edges, 'neighbors_row_splits': 10 split points}
      NEW:  {'neighbors_index': 6 edges, 'segment_ids': 6 ids, 'counts': 9 counts}
      Sample dict keys after: ['neighbors_index', 'segment_ids', 'counts']
```

---

## SAMPLE 0 - neighbors_out Processing

```
>>> LINE 131: nbrs = s[key]
    OUTPUT: nbrs dict with 2 keys: ['neighbors_index', 'neighbors_row_splits']

>>> LINE 132: idx = np.asarray(nbrs['neighbors_index'])
    OUTPUT: idx = [6, 5, 3, 5, 2, 5, 1, 2, 7, 0, 0, 6, 1]
            shape = (13,), dtype = int32

>>> LINE 133: splits = np.asarray(nbrs['neighbors_row_splits'])
    OUTPUT: splits = [0, 2, 4, 7, 10, 13]
            shape = (6,), dtype = int32
            (5 vertices + 1 = 6 split points)

>>> LINE 134: n_real = idx.shape[0]
    OUTPUT: n_real = 13 (actual edges in this sample)

>>> LINE 135: n_out = splits.shape[0] - 1
    OUTPUT: n_out = 5 (number of output nodes)
            (5 vertices = 5 output nodes)

>>> LINE 137: counts = splits[1:] - splits[:-1]
    OUTPUT: counts = [2, 2, 3, 3, 3]
            (edges per vertex)

>>> LINE 138: seg_ids = np.repeat(np.arange(n_out), counts)
    OUTPUT: seg_ids = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
            (each edge gets vertex ID: vertex 0 appears in 2 edges, vertex 1 in 2, etc.)

>>> LINE 139: pad_len = max_edges - n_real
    OUTPUT: pad_len = 13 - 13 = 0
            (No padding needed - this sample has max edges)

>>> LINE 141: idx_padded = np.concatenate([idx, np.zeros(pad_len, ...)])
    OUTPUT: idx_padded = [6, 5, 3, 5, 2, 5, 1, 2, 7, 0, 0, 6, 1]
            shape = (13,)

>>> LINE 142: seg_ids_padded = np.concatenate([seg_ids, np.full(pad_len, n_out, ...)])
    OUTPUT: seg_ids_padded = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
            shape = (13,)
            (last 0 values are dummy segment 5)

>>> LINE 151-155: s['neighbors_out'] = { updated_dict }
    OUTPUT: Updated s['neighbors_out'] structure:
      OLD:  {'neighbors_index': 13 edges, 'neighbors_row_splits': 6 split points}
      NEW:  {'neighbors_index': 13 edges, 'segment_ids': 13 ids, 'counts': 5 counts}
      Sample dict keys after: ['neighbors_index', 'segment_ids', 'counts']
```

---

## SAMPLES 1-6: Pattern Identical to SAMPLE 0

All remaining samples (1, 2, 3, 4, 5, 6) follow the **exact same pattern**:

- **neighbors_in:** 6 edges → padded to 6 edges (no padding needed)
- **neighbors_out:** 13 edges → padded to 13 edges (no padding needed)

Structure transformation for each:

**OLD Structure (2 keys):**
```python
neighbors_in = {
    'neighbors_index': [6 edges],          # e.g., [2, 0, 0, 4, 2, 2]
    'neighbors_row_splits': [10 values],   # [0, 0, 1, 1, 2, 3, 4, 5, 5, 6]
}

neighbors_out = {
    'neighbors_index': [13 edges],         # e.g., [3, 4, 3, 6, 1, 2, 0, 3, 2, 1, 7, 4, 4]
    'neighbors_row_splits': [6 values],    # [0, 2, 4, 7, 10, 13]
}
```

**NEW Structure (3 keys):**
```python
neighbors_in = {
    'neighbors_index': [6 edges],          # e.g., [2, 0, 0, 4, 2, 2]
    'segment_ids': [6 ids],                # e.g., [1, 3, 4, 5, 6, 8]  (maps edges to nodes 0-8)
    'counts': [9 values],                  # [0, 1, 0, 1, 1, 1, 1, 0, 1]  (edge counts per node)
}

neighbors_out = {
    'neighbors_index': [13 edges],         # e.g., [3, 4, 3, 6, 1, 2, 0, 3, 2, 1, 7, 4, 4]
    'segment_ids': [13 ids],               # [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]  (maps to vertices 0-4)
    'counts': [5 values],                  # [2, 2, 3, 3, 3]  (edge counts per vertex)
}
```

---

## LINE 158: Return Statement

```
>>> LINE 158: return max_in, max_out
    RETURN VALUE: (6, 13)
```

---

## Final State After Execution

```
================================================================================
EXECUTION COMPLETE
================================================================================

Final Return: max_in=6, max_out=13

Key Insights:
  - All 7 samples padded to uniform shape: (6, 13)
  - Enables JIT compilation with fixed tensor shapes
  - Dummy segments (n_out) are safely ignored during aggregation
```

---

## Data Transformation Summary

| Aspect | Before | After |
|--------|--------|-------|
| **neighbors_in keys** | 2 (index, row_splits) | 3 (index, segment_ids, counts) |
| **neighbors_out keys** | 2 (index, row_splits) | 3 (index, segment_ids, counts) |
| **neighbors_in shape** | (6,) variable | (6,) fixed |
| **neighbors_out shape** | (13,) variable | (13,) fixed |
| **Purpose** | Raw sparse representation | JIT-compatible dense representation |

---

## Why This Matters

1. **Fixed Shapes:** All samples now have fixed tensor shapes (6 and 13), enabling XLA/JIT compilation
2. **Segment IDs:** Map each edge to its source node for aggregation operations (mean/sum pooling)
3. **Counts:** Track real vs. padded edges to correctly normalize during reduction
4. **Dummy Padding:** Padding with segment ID `n_out` ensures dummy edges are ignored in aggregation

Example aggregation:
```
For neighbors_out with segment_ids [0,0,1,1,2,2,2,3,3,3,4,4,4]:
  - Edges 0-1 belong to segment 0 → will be pooled together
  - Edges 2-3 belong to segment 1 → will be pooled together
  - Edges 4-6 belong to segment 2 → will be pooled together
  - Edges 7-9 belong to segment 3 → will be pooled together
  - Edges 10-12 belong to segment 4 → will be pooled together
  
Any padding edges with segment_id=5 are safely ignored
```
