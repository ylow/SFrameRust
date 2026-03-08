use pyo3::prelude::*;

/// Runtime configuration for the SFrame engine.
///
/// Access as `sframe.config`. All numeric settings can be read and written
/// as Python attributes. `cache_dir` is read-only.
///
///     >>> import sframe
///     >>> sframe.config.cache_capacity
///     2147483648
///     >>> sframe.config.cache_capacity = 4 * 1024**3
///     >>> sframe.config.reset()  # restore defaults
///
#[pyclass(name = "Config")]
pub struct PyConfig;

#[pymethods]
impl PyConfig {
    // -- cache_dir (read-only) --

    #[getter]
    fn cache_dir(&self) -> String {
        sframe_config::global()
            .cache_dir
            .to_string_lossy()
            .into_owned()
    }

    // -- cache_capacity --

    #[getter]
    fn cache_capacity(&self) -> usize {
        sframe_config::global().cache_capacity()
    }

    #[setter]
    fn set_cache_capacity(&self, value: usize) {
        sframe_config::global().set_cache_capacity(value);
    }

    // -- cache_capacity_per_file --

    #[getter]
    fn cache_capacity_per_file(&self) -> usize {
        sframe_config::global().cache_capacity_per_file()
    }

    #[setter]
    fn set_cache_capacity_per_file(&self, value: usize) {
        sframe_config::global().set_cache_capacity_per_file(value);
    }

    // -- source_batch_size --

    #[getter]
    fn source_batch_size(&self) -> usize {
        sframe_config::global().source_batch_size()
    }

    #[setter]
    fn set_source_batch_size(&self, value: usize) {
        sframe_config::global().set_source_batch_size(value);
    }

    // -- sort_max_memory --

    #[getter]
    fn sort_max_memory(&self) -> usize {
        sframe_config::global().sort_max_memory()
    }

    #[setter]
    fn set_sort_max_memory(&self, value: usize) {
        sframe_config::global().set_sort_max_memory(value);
    }

    // -- groupby_buffer_num_rows --

    #[getter]
    fn groupby_buffer_num_rows(&self) -> usize {
        sframe_config::global().groupby_buffer_num_rows()
    }

    #[setter]
    fn set_groupby_buffer_num_rows(&self, value: usize) {
        sframe_config::global().set_groupby_buffer_num_rows(value);
    }

    // -- join_buffer_num_cells --

    #[getter]
    fn join_buffer_num_cells(&self) -> usize {
        sframe_config::global().join_buffer_num_cells()
    }

    #[setter]
    fn set_join_buffer_num_cells(&self, value: usize) {
        sframe_config::global().set_join_buffer_num_cells(value);
    }

    // -- source_prefetch_segments --

    #[getter]
    fn source_prefetch_segments(&self) -> usize {
        sframe_config::global().source_prefetch_segments()
    }

    #[setter]
    fn set_source_prefetch_segments(&self, value: usize) {
        sframe_config::global().set_source_prefetch_segments(value);
    }

    // -- max_blocks_in_cache --

    #[getter]
    fn max_blocks_in_cache(&self) -> usize {
        sframe_config::global().max_blocks_in_cache()
    }

    #[setter]
    fn set_max_blocks_in_cache(&self, value: usize) {
        sframe_config::global().set_max_blocks_in_cache(value);
    }

    // -- reset --

    /// Reset all mutable settings to their defaults (or environment variable
    /// overrides).
    fn reset(&self) {
        sframe_config::global().reset();
    }

    // -- __repr__ / __str__ --

    fn __repr__(&self) -> String {
        let cfg = sframe_config::global();
        format!(
            "Config(\n\
             \x20 cache_dir={:?},\n\
             \x20 cache_capacity={},\n\
             \x20 cache_capacity_per_file={},\n\
             \x20 source_batch_size={},\n\
             \x20 sort_max_memory={},\n\
             \x20 groupby_buffer_num_rows={},\n\
             \x20 join_buffer_num_cells={},\n\
             \x20 source_prefetch_segments={},\n\
             \x20 max_blocks_in_cache={},\n\
             )",
            cfg.cache_dir.display(),
            cfg.cache_capacity(),
            cfg.cache_capacity_per_file(),
            cfg.source_batch_size(),
            cfg.sort_max_memory(),
            cfg.groupby_buffer_num_rows(),
            cfg.join_buffer_num_cells(),
            cfg.source_prefetch_segments(),
            cfg.max_blocks_in_cache(),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
