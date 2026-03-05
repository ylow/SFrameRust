# compile_stream Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate materialization fallbacks from `compile_stream()` and related methods by making `fuse_plan()` return `Result` and routing all execution through the plan executor.

**Architecture:** Change `fuse_plan()` from `Option<Arc<PlannerNode>>` to `Result<Arc<PlannerNode>>`, then simplify `compile_stream()`, `materialize_batch()`, and 4 operations (`filter`, `logical_filter`, `append`, `sample`) to remove their materialization fallbacks.

**Tech Stack:** Rust, sframe-query executor, PlannerNode, ColumnUnion

---

### Task 1: Make `fuse_plan()` return `Result`

**Files:**
- Modify: `crates/sframe/src/sframe.rs:1594-1663` (`fuse_plan`)

**Step 1: Change the signature and error paths**

Change `fuse_plan` from `fn fuse_plan(&self) -> Option<Arc<PlannerNode>>` to `fn fuse_plan(&self) -> Result<Arc<PlannerNode>>`.

Three changes inside the body:
1. Empty columns (line 1595-1596): return a materialized empty plan instead of `None`
2. Incompatible lengths (line 1645-1646): return an error instead of `None`
3. All `Some(...)` return values become `Ok(...)`

```rust
fn fuse_plan(&self) -> Result<Arc<PlannerNode>> {
    if self.columns.is_empty() {
        return Ok(PlannerNode::materialized(SFrameRows::empty(&[])));
    }

    // Fast path: all columns share the same plan Arc — just project.
    if let Some(plan) = self.shared_plan() {
        let indices: Vec<usize> =
            self.columns.iter().map(|c| c.column_index()).collect();
        return Ok(PlannerNode::project(plan.clone(), indices));
    }

    // Group columns by plan Arc, preserving original order.
    use std::collections::HashMap;
    let mut plan_to_group: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<(Arc<PlannerNode>, Vec<usize>)> = Vec::new();
    let mut col_positions: Vec<(usize, usize)> = Vec::with_capacity(self.columns.len());

    for col in &self.columns {
        let ptr = Arc::as_ptr(col.plan()) as usize;
        let group_idx = if let Some(&gi) = plan_to_group.get(&ptr) {
            gi
        } else {
            let gi = groups.len();
            plan_to_group.insert(ptr, gi);
            groups.push((col.plan().clone(), Vec::new()));
            gi
        };
        let pos = groups[group_idx].1.len();
        groups[group_idx].1.push(col.column_index());
        col_positions.push((group_idx, pos));
    }

    // Build Project for each group
    let mut union_inputs: Vec<Arc<PlannerNode>> = Vec::new();
    let mut group_col_offsets: Vec<usize> = Vec::new();
    let mut offset = 0;
    for (plan, indices) in &groups {
        group_col_offsets.push(offset);
        offset += indices.len();
        union_inputs.push(PlannerNode::project(plan.clone(), indices.clone()));
    }

    let fused = if union_inputs.len() == 1 {
        union_inputs.remove(0)
    } else {
        // Verify length compatibility
        let known_lengths: Vec<u64> = union_inputs.iter()
            .filter_map(|p| p.length())
            .collect();
        if known_lengths.len() >= 2 && !known_lengths.windows(2).all(|w| w[0] == w[1]) {
            return Err(SFrameError::Format(
                "Cannot fuse columns with different row counts".to_string()
            ));
        }
        PlannerNode::column_union(union_inputs)
    };

    // Build final column order
    let final_indices: Vec<usize> = col_positions.iter()
        .map(|&(gi, pos)| group_col_offsets[gi] + pos)
        .collect();

    let is_identity = final_indices.iter().enumerate().all(|(i, &c)| c == i);
    if is_identity {
        Ok(fused)
    } else {
        Ok(PlannerNode::project(fused, final_indices))
    }
}
```

**Step 2: Fix all callers of `fuse_plan()` that use `Option` matching**

Every caller that does `if let Some(fused) = self.fuse_plan()` must change. This step just fixes the compilation — callers are updated in subsequent tasks.

Callers to update (these will be detailed in Tasks 2-5):
- `compile_stream()` (line 1675)
- `materialize_batch()` (line 1697)
- `filter()` (line 578)
- `logical_filter()` (line 626)
- `append()` (line 681)
- `sample()` (line 1057)
- `add_column()` (line 516)
- `replace_column()` (line 882)
- `explain()` (line 458)

For `add_column`, `replace_column`, and `explain` — keep current behavior but adapt to `Result`:
- `add_column`: change `if let Some(fused) = self.fuse_plan()` to `if let Ok(fused) = self.fuse_plan()` (keep fallback to independent plans, it's not materialization)
- `replace_column`: same — `if let Ok(fused) = self.fuse_plan()` (keep fallback)
- `explain`: change `if let Some(fused) = self.fuse_plan()` to `if let Ok(fused) = self.fuse_plan()`

**Step 3: Run tests**

Run: `cargo test -p sframe 2>&1 | tail -5`
Expected: All tests pass

**Step 4: Commit**

```
git add crates/sframe/src/sframe.rs
git commit -m "refactor: make fuse_plan() return Result instead of Option"
```

---

### Task 2: Simplify `compile_stream()` and `materialize_batch()`

**Files:**
- Modify: `crates/sframe/src/sframe.rs:1669-1709`

**Step 1: Replace `compile_stream()`**

Replace the current 14-line method with:

```rust
/// Compile the SFrame's plan into a BatchStream.
pub(crate) fn compile_stream(&self) -> Result<sframe_query::execute::BatchStream> {
    compile(&self.fuse_plan()?)
}
```

**Step 2: Replace `materialize_batch()`**

Replace the current 17-line method with:

```rust
fn materialize_batch(&self) -> Result<SFrameRows> {
    materialize_sync(self.compile_stream()?)
}
```

**Step 3: Run tests**

Run: `cargo test -p sframe 2>&1 | tail -5`
Expected: All tests pass

**Step 4: Commit**

```
git add crates/sframe/src/sframe.rs
git commit -m "refactor: simplify compile_stream and materialize_batch to use fuse_plan directly"
```

---

### Task 3: Remove materialization fallback from `filter()`

**Files:**
- Modify: `crates/sframe/src/sframe.rs:570-612`

**Step 1: Replace `filter()` body**

Remove the `if let` / fallback pattern. Use `fuse_plan()?` directly:

```rust
pub fn filter(
    &self,
    column_name: &str,
    pred: Arc<dyn Fn(&FlexType) -> bool + Send + Sync>,
) -> Result<SFrame> {
    let filter_col_idx = self.column_index(column_name)?;
    let fused = self.fuse_plan()?;

    let mask_source = PlannerNode::project(fused.clone(), vec![filter_col_idx]);
    let mask = PlannerNode::transform(
        mask_source,
        0,
        Arc::new(move |v: &FlexType| -> FlexType {
            if pred(v) {
                FlexType::Integer(1)
            } else {
                FlexType::Integer(0)
            }
        }),
        FlexTypeEnum::Integer,
    );

    let filtered_plan = PlannerNode::logical_filter(fused, mask);

    let columns: Vec<SArray> = self
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| SArray::from_plan(filtered_plan.clone(), c.dtype(), None, i))
        .collect();

    Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
}
```

**Step 2: Remove materialization fallback from `logical_filter()`**

Same pattern — lines 625-667 become:

```rust
pub fn logical_filter(&self, mask: SArray) -> Result<SFrame> {
    let fused = self.fuse_plan()?;
    let filtered_plan = PlannerNode::logical_filter(fused, mask.plan().clone());

    let columns: Vec<SArray> = self
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| SArray::from_plan(filtered_plan.clone(), c.dtype(), None, i))
        .collect();

    Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
}
```

**Step 3: Run tests**

Run: `cargo test -p sframe 2>&1 | tail -5`
Expected: All tests pass

**Step 4: Commit**

```
git add crates/sframe/src/sframe.rs
git commit -m "refactor: remove materialization fallback from filter and logical_filter"
```

---

### Task 4: Remove materialization fallback from `append()`

**Files:**
- Modify: `crates/sframe/src/sframe.rs:673-699`

**Step 1: Replace `append()` body**

```rust
pub fn append(&self, other: &SFrame) -> Result<SFrame> {
    if self.column_names != other.column_names {
        return Err(SFrameError::Format(
            "Column names must match for append".to_string(),
        ));
    }

    let left = self.fuse_plan()?;
    let right = other.fuse_plan()?;
    let appended = PlannerNode::append(left, right);

    let columns: Vec<SArray> = self
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| SArray::from_plan(appended.clone(), c.dtype(), None, i))
        .collect();

    Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
}
```

**Step 2: Run tests**

Run: `cargo test -p sframe 2>&1 | tail -5`
Expected: All tests pass

**Step 3: Commit**

```
git add crates/sframe/src/sframe.rs
git commit -m "refactor: remove materialization fallback from append"
```

---

### Task 5: Remove materialization fallback from `sample()`

**Files:**
- Modify: `crates/sframe/src/sframe.rs:1031-1086`

**Step 1: Replace `sample()` body (after the `make_mask` closure)**

Replace lines 1057-1085 with:

```rust
    let fused = self.fuse_plan()?;
    let mask = make_mask();
    let filtered = PlannerNode::logical_filter(fused, mask);
    let columns: Vec<SArray> = self
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| SArray::from_plan(filtered.clone(), c.dtype(), None, i))
        .collect();
    Ok(SFrame::new_with_columns(columns, self.column_names.clone()))
```

**Step 2: Run tests**

Run: `cargo test -p sframe 2>&1 | tail -5`
Expected: All tests pass

**Step 3: Commit**

```
git add crates/sframe/src/sframe.rs
git commit -m "refactor: remove materialization fallback from sample"
```

---

### Task 6: Full test suite + clippy verification

**Files:** None (verification only)

**Step 1: Run full workspace tests**

Run: `cargo test --workspace 2>&1 | tail -5`
Expected: All tests pass, 0 failures

**Step 2: Run clippy**

Run: `cargo clippy --workspace 2>&1 | grep -E "warning|error" | grep -v "generated" | head -20`
Expected: No new warnings from the changed code

**Step 3: Verify dead code removal**

Check that no unused imports remain from removed materialization fallback code (e.g., `filter_by_column`, `from_column_vecs` if they were only used in removed paths). If any unused import warnings appear, fix them.

**Step 4: Commit any cleanup**

Only if step 3 found issues:
```
git add -u
git commit -m "chore: remove unused imports from materialization cleanup"
```
