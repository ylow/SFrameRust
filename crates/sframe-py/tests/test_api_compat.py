"""Tests for C++ API compatibility additions (Phase 1 & 2)."""

import pytest
from _sframe import SFrame, SArray, Sketch, aggregate


# ═══════════════════════════════════════════════════════════════════
# SFrame trivial additions
# ═══════════════════════════════════════════════════════════════════


class TestSFrameShapeDtype:
    def test_shape(self):
        sf = SFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert sf.shape == (3, 2)

    def test_dtype_property(self):
        sf = SFrame({"x": [1, 2], "y": [1.0, 2.0], "z": ["a", "b"]})
        dt = sf.dtype
        assert dt["x"] == "int"
        assert dt["y"] == "float"
        assert dt["z"] == "str"

    def test_num_cols(self):
        sf = SFrame({"a": [1], "b": [2], "c": [3]})
        assert sf.num_cols() == 3
        assert sf.num_cols() == sf.num_columns()


class TestSFrameAliases:
    def test_select_columns(self):
        sf = SFrame({"a": [1], "b": [2], "c": [3]})
        sf2 = sf.select_columns(["a", "c"])
        assert sf2.column_names() == ["a", "c"]

    def test_select_column(self):
        sf = SFrame({"a": [10, 20]})
        sa = sf.select_column("a")
        assert sa.to_list() == [10, 20]

    def test_read_csv_alias(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        sf = SFrame.read_csv(str(csv_file))
        assert sf.num_rows() == 2

    def test_read_json_alias(self, tmp_path):
        json_file = tmp_path / "test.json"
        json_file.write_text('{"a": 1}\n{"a": 2}\n')
        sf = SFrame.read_json(str(json_file))
        assert sf.num_rows() == 2

    def test_export_csv_alias(self, tmp_path):
        sf = SFrame({"a": [1, 2]})
        out = str(tmp_path / "out.csv")
        sf.export_csv(out)
        content = open(out).read()
        assert "a" in content

    def test_export_json_alias(self, tmp_path):
        sf = SFrame({"a": [1, 2]})
        out = str(tmp_path / "out.json")
        sf.export_json(out)
        import json
        lines = [json.loads(line) for line in open(out) if line.strip()]
        assert len(lines) == 2


class TestSFrameDelitemCopy:
    def test_delitem(self):
        sf = SFrame({"a": [1, 2], "b": [3, 4]})
        del sf["b"]
        assert sf.column_names() == ["a"]

    def test_copy(self):
        sf = SFrame({"a": [1, 2]})
        sf2 = sf.copy()
        assert sf2.num_rows() == 2
        assert sf2.column_names() == ["a"]


# ═══════════════════════════════════════════════════════════════════
# SFrame constructor
# ═══════════════════════════════════════════════════════════════════


class TestSFrameConstructor:
    def test_from_dict_of_lists(self):
        sf = SFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert sf.num_rows() == 3
        assert sf.num_columns() == 2

    def test_from_dict_of_sarrays(self):
        sa = SArray([10, 20, 30])
        sf = SFrame({"col": sa})
        assert sf.num_rows() == 3
        assert sf.column("col").to_list() == [10, 20, 30]

    def test_from_list_of_dicts(self):
        sf = SFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert sf.num_rows() == 2
        assert set(sf.column_names()) == {"a", "b"}

    def test_from_list_of_dicts_ragged(self):
        sf = SFrame([{"a": 1}, {"a": 2, "b": 3}])
        assert sf.num_rows() == 2
        # First row should have None/Undefined for b
        row0 = sf[0]
        assert row0["b"] is None

    def test_empty_constructor(self):
        sf = SFrame()
        assert sf.num_rows() == 0
        assert sf.num_columns() == 0

    def test_none_constructor(self):
        sf = SFrame(None)
        assert sf.num_rows() == 0


# ═══════════════════════════════════════════════════════════════════
# SFrame easy additions
# ═══════════════════════════════════════════════════════════════════


class TestSFrameAddRemoveColumns:
    def test_add_columns_dict(self):
        sf = SFrame({"a": [1, 2]})
        sf2 = sf.add_columns({"b": SArray([3, 4]), "c": SArray([5, 6])})
        assert set(sf2.column_names()) == {"a", "b", "c"}

    def test_remove_columns(self):
        sf = SFrame({"a": [1], "b": [2], "c": [3]})
        sf2 = sf.remove_columns(["a", "c"])
        assert sf2.column_names() == ["b"]


class TestSFrameFilterBy:
    def test_filter_by_include(self):
        sf = SFrame({"a": [1, 2, 3, 4, 5]})
        sf2 = sf.filter_by([2, 4], "a")
        vals = sorted(sf2.column("a").to_list())
        assert vals == [2, 4]

    def test_filter_by_exclude(self):
        sf = SFrame({"a": [1, 2, 3, 4, 5]})
        sf2 = sf.filter_by([2, 4], "a", exclude=True)
        vals = sorted(sf2.column("a").to_list())
        assert vals == [1, 3, 5]

    def test_filter_by_sarray(self):
        sf = SFrame({"a": [1, 2, 3, 4, 5]})
        sf2 = sf.filter_by(SArray([1, 5]), "a")
        vals = sorted(sf2.column("a").to_list())
        assert vals == [1, 5]


class TestSFrameAddRowNumber:
    def test_basic(self):
        sf = SFrame({"a": ["x", "y", "z"]})
        sf2 = sf.add_row_number()
        assert sf2.column_names()[0] == "id"
        assert sf2.column("id").to_list() == [0, 1, 2]

    def test_custom_name_start(self):
        sf = SFrame({"a": [10, 20]})
        sf2 = sf.add_row_number("row_id", start=5)
        assert sf2.column("row_id").to_list() == [5, 6]
        assert sf2.column_names()[0] == "row_id"


class TestSFrameDropnaSplit:
    def test_basic_split(self):
        sf = SFrame({"a": [1, None, 3], "b": [4, 5, None]})
        clean, dirty = sf.dropna_split()
        assert clean.num_rows() == 1
        assert dirty.num_rows() == 2


class TestSFramePrintRows:
    def test_basic(self):
        sf = SFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = sf.print_rows()
        assert "a" in result
        assert "b" in result


# ═══════════════════════════════════════════════════════════════════
# SArray trivial additions
# ═══════════════════════════════════════════════════════════════════


class TestSArrayShapeSize:
    def test_shape(self):
        sa = SArray([1, 2, 3])
        assert sa.shape == (3,)

    def test_size(self):
        sa = SArray([1, 2, 3])
        assert sa.size() == 3


class TestSArrayClipVariants:
    def test_clip_lower(self):
        sa = SArray([1, 5, 10])
        result = sa.clip_lower(3).to_list()
        assert result == [3, 5, 10]

    def test_clip_upper(self):
        sa = SArray([1, 5, 10])
        result = sa.clip_upper(7).to_list()
        assert result == [1, 5, 7]


class TestSArrayUnaryOps:
    def test_neg(self):
        sa = SArray([1, -2, 3])
        result = (-sa).to_list()
        assert result == [-1, 2, -3]

    def test_pos(self):
        sa = SArray([1, -2, 3])
        result = (+sa).to_list()
        assert result == [1, -2, 3]

    def test_abs(self):
        sa = SArray([-1, 2, -3])
        result = abs(sa).to_list()
        assert result == [1, 2, 3]


class TestSArrayFloorDiv:
    def test_scalar(self):
        sa = SArray([7, 10, 15])
        result = (sa // 3).to_list()
        assert result == [2, 3, 5]

    def test_float_scalar(self):
        sa = SArray([7.0, 10.0, 15.5])
        result = (sa // 3.0).to_list()
        assert result == [2.0, 3.0, 5.0]

    def test_rfloordiv(self):
        sa = SArray([2, 3, 4])
        result = (10 // sa).to_list()
        assert result == [5, 3, 2]


class TestSArrayPow:
    def test_scalar(self):
        sa = SArray([2, 3, 4])
        result = (sa ** 2).to_list()
        assert result == [4.0, 9.0, 16.0]

    def test_rpow(self):
        sa = SArray([1, 2, 3])
        result = (2 ** sa).to_list()
        assert result == [2.0, 4.0, 8.0]


# ═══════════════════════════════════════════════════════════════════
# SArray easy additions
# ═══════════════════════════════════════════════════════════════════


class TestSArrayContains:
    def test_contains_true(self):
        sa = SArray([1, 2, 3])
        assert 2 in sa

    def test_contains_false(self):
        sa = SArray([1, 2, 3])
        assert 5 not in sa


class TestSArrayIsIn:
    def test_basic(self):
        sa = SArray([1, 2, 3, 4, 5])
        result = sa.is_in([2, 4]).to_list()
        assert result == [0, 1, 0, 1, 0]

    def test_with_sarray(self):
        sa = SArray([1, 2, 3, 4, 5])
        lookup = SArray([1, 5])
        result = sa.is_in(lookup).to_list()
        assert result == [1, 0, 0, 0, 1]


class TestSArrayFromConst:
    def test_int_const(self):
        sa = SArray.from_const(42, 5)
        assert sa.to_list() == [42, 42, 42, 42, 42]
        assert sa.dtype == "int"

    def test_str_const(self):
        sa = SArray.from_const("hello", 3)
        assert sa.to_list() == ["hello", "hello", "hello"]
        assert sa.dtype == "str"


class TestSArrayRandomSplit:
    def test_basic(self):
        sa = SArray(list(range(100)))
        a, b = sa.random_split(0.8, seed=42)
        assert len(a) + len(b) == 100


# ═══════════════════════════════════════════════════════════════════
# Aggregate aliases
# ═══════════════════════════════════════════════════════════════════


class TestAggregateAliases:
    def test_avg(self):
        sf = SFrame({"k": ["a", "a", "b"], "v": [1, 3, 10]})
        result = sf.groupby(["k"], {"avg_v": aggregate.AVG("v")})
        vals = {row["k"]: row["avg_v"] for row in result}
        assert vals["a"] == 2.0
        assert vals["b"] == 10.0

    def test_stdv(self):
        spec = aggregate.STDV("v")
        assert repr(spec) == "aggregate.STD('v')"


# ═══════════════════════════════════════════════════════════════════
# Sketch
# ═══════════════════════════════════════════════════════════════════


class TestSketchNumeric:
    def test_exact_stats(self):
        sa = SArray([1, 2, 3, 4, 5])
        s = sa.sketch_summary()
        assert s.size() == 5
        assert s.min() == 1
        assert s.max() == 5
        assert s.mean() == 3.0
        assert s.sum() == 15
        assert s.num_undefined() == 0

    def test_variance_std(self):
        sa = SArray([2, 4, 4, 4, 5, 5, 7, 9])
        s = sa.sketch_summary()
        assert abs(s.mean() - 5.0) < 1e-10
        assert abs(s.var() - 4.571428571428571) < 1e-10
        assert abs(s.std() - s.var() ** 0.5) < 1e-10

    def test_quantiles(self):
        sa = SArray(list(range(1, 101)))
        s = sa.sketch_summary()
        assert s.quantile(0.0) == 1
        assert s.quantile(1.0) == 100
        q50 = s.quantile(0.5)
        assert 45 <= q50 <= 55  # approximate

    def test_num_unique(self):
        sa = SArray([1, 1, 2, 2, 3, 3, 4, 5])
        s = sa.sketch_summary()
        # HyperLogLog is approximate, but for small sets should be exact
        assert 4 <= s.num_unique() <= 6

    def test_frequent_items(self):
        sa = SArray([1, 1, 1, 2, 2, 3])
        s = sa.sketch_summary()
        freq = s.frequent_items()
        assert freq[1] == 3
        assert freq[2] == 2

    def test_frequency_count(self):
        sa = SArray([1, 1, 1, 2, 2, 3])
        s = sa.sketch_summary()
        assert s.frequency_count(1) == 3
        assert s.frequency_count(2) == 2
        assert s.frequency_count(99) == 0


class TestSketchString:
    def test_basic(self):
        sa = SArray(["a", "b", "a", "c"])
        s = sa.sketch_summary()
        assert s.size() == 4
        assert s.num_undefined() == 0
        assert s.num_unique() >= 3
        freq = s.frequent_items()
        assert freq["a"] == 2

    def test_with_missing(self):
        sa = SArray(["a", None, "b", None])
        s = sa.sketch_summary()
        assert s.size() == 4
        assert s.num_undefined() == 2


class TestSketchMissing:
    def test_numeric_with_none(self):
        sa = SArray([1, None, 3, None, 5])
        s = sa.sketch_summary()
        assert s.size() == 5
        assert s.num_undefined() == 2
        assert s.min() == 1
        assert s.max() == 5
        assert s.mean() == 3.0  # mean of [1, 3, 5]

    def test_empty(self):
        sa = SArray([])
        s = sa.sketch_summary()
        assert s.size() == 0
        assert s.num_undefined() == 0


class TestSketchConstructor:
    def test_via_class(self):
        s = Sketch(SArray([10, 20, 30]))
        assert s.size() == 3
        assert s.sum() == 60

    def test_repr(self):
        s = Sketch(SArray([1, 2, 3]))
        r = repr(s)
        assert "Length" in r
        assert "Min" in r
        assert "Quantiles" in r


class TestSketchFloat:
    def test_float_array(self):
        sa = SArray([1.5, 2.5, 3.5])
        s = sa.sketch_summary()
        assert s.min() == 1.5
        assert s.max() == 3.5
        assert abs(s.mean() - 2.5) < 1e-10
        assert abs(s.sum() - 7.5) < 1e-10
