# Design: Eliminate Materialization Fallbacks from compile_stream

## Problem

`compile_stream()` turns an SFrame into a `BatchStream`. With the new
ColumnUnion-based `fuse_plan()`, fusion almost never fails — it only returns
`None` for empty columns (trivial) or groups with statically incompatible
lengths (a real error). Yet `compile_stream()` still has a full-materialization
fallback, and the same pattern repeats across 6+ methods.

## Fix

### 1. Make `fuse_plan()` return `Result<Arc<PlannerNode>>` instead of `Option`

- Empty columns → return a materialized empty plan
- Incompatible lengths → return an error
- Otherwise → always succeed via ColumnUnion

### 2. Simplify `compile_stream()` and `materialize_batch()`

```rust
fn compile_stream(&self) -> Result<BatchStream> {
    compile(&self.fuse_plan()?)
}

fn materialize_batch(&self) -> Result<SFrameRows> {
    materialize_sync(self.compile_stream()?)
}
```

No fallbacks. All roads go through the executor.

### 3. Remove materialization fallbacks from lazy operations

- `filter()`, `logical_filter()`, `sample()` — replace
  `if let Some(fused)` with `let fused = self.fuse_plan()?;`, delete the
  materialize fallback branch.
- `append()` — same for both sides.

### 4. Leave `add_column()` / `replace_column()` fallbacks alone

Their fallback creates independent plans (not materialization). `fuse_plan()`
handles fusion via ColumnUnion when the SFrame is later consumed.

## Out of Scope

Methods that unconditionally call `materialize_batch()` without even trying
fusion (`dropna`, `pack_columns`, `unpack_column`, `stack`, `unique`,
`random_split`, `to_json`, `iter_rows`) — separate concern.
