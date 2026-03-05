# Design: Block-Level Source Streaming

## Problem

The source reader reads entire segments (1M rows) into memory before yielding
any batches. For `head(10)` through a filter, the background thread reads 2M+
rows just to return 10 rows.

## Fix

Change the source's background thread from segment-level to batch-level
reading. Instead of reading a full segment, read in chunks of `batch_size`
rows using the existing `read_columns_block_range` which does block-level
seeking.

### Background thread

Before: read entire segment, send through channel.
After: open SegmentReader once per segment, read chunks of ~batch_size rows
via `read_columns_block_range`, send each chunk through channel.

### unfold_prefetch

Simplifies — chunks are already batch-sized, so no more `seg_data` slicing.
Just receive chunk, convert to `SFrameRows`, yield.

### Channel capacity

Changes from "2 segments ahead" (2M rows) to "N batches ahead" (~4-8
batches). The `source_prefetch_segments` config is repurposed to control
prefetch batch count.

### Projected reads

`compile_sframe_source_projected` gets the same treatment — chunked projected
reads via `read_columns_block_range` with `column_indices`.

### What stays the same

- `SegmentReader` API unchanged
- `read_columns_block_range` already exists
- Stream interface (`BatchStream`) unchanged
- All consumers unchanged
