"""Basic integration tests for the sframe Python bindings."""

import os
import tempfile

import _sframe
from _sframe import SFrame, SArray, SFrameStreamWriter, aggregate, load


class TestSArray:
    def test_construct_int(self):
        sa = SArray([1, 2, 3])
        assert sa.dtype == "int"
        assert len(sa) == 3
        assert sa.head(3) == [1, 2, 3]

    def test_construct_float(self):
        sa = SArray([1.0, 2.5, 3.7])
        assert sa.dtype == "float"
        assert len(sa) == 3

    def test_construct_string(self):
        sa = SArray(["a", "b", "c"])
        assert sa.dtype == "str"
        assert sa.to_list() == ["a", "b", "c"]

    def test_construct_with_dtype(self):
        sa = SArray([1, 2, 3], dtype="float")
        assert sa.dtype == "float"

    def test_arithmetic_scalar(self):
        sa = SArray([1, 2, 3])
        result = sa + 10
        assert result.to_list() == [11, 12, 13]

    def test_arithmetic_array(self):
        a = SArray([1, 2, 3])
        b = SArray([10, 20, 30])
        result = a + b
        assert result.to_list() == [11, 22, 33]

    def test_comparison(self):
        sa = SArray([1, 2, 3, 4, 5])
        mask = sa > 3
        assert mask.dtype == "int"
        assert mask.to_list() == [0, 0, 0, 1, 1]

    def test_sum(self):
        sa = SArray([1, 2, 3, 4, 5])
        assert sa.sum() == 15

    def test_mean(self):
        sa = SArray([2.0, 4.0, 6.0])
        assert sa.mean() == 4.0

    def test_min_max(self):
        sa = SArray([3, 1, 4, 1, 5])
        assert sa.min() == 1
        assert sa.max() == 5

    def test_any_all(self):
        sa = SArray([0, 0, 0])
        assert not sa.any()
        assert not sa.all()

        sa2 = SArray([1, 0, 1])
        assert sa2.any()
        assert not sa2.all()

        sa3 = SArray([1, 1, 1])
        assert sa3.any()
        assert sa3.all()

    def test_dropna(self):
        sa = SArray([1, None, 3, None, 5])
        result = sa.dropna()
        assert result.to_list() == [1, 3, 5]

    def test_fillna(self):
        sa = SArray([1, None, 3])
        result = sa.fillna(0)
        assert result.to_list() == [1, 0, 3]

    def test_is_na(self):
        sa = SArray([1, None, 3])
        result = sa.is_na()
        assert result.to_list() == [0, 1, 0]

    def test_apply(self):
        sa = SArray([1, 2, 3])
        result = sa.apply(lambda x: x * 2, dtype="int")
        assert result.to_list() == [2, 4, 6]

    def test_filter(self):
        sa = SArray([1, 2, 3, 4, 5])
        result = sa.filter(lambda x: x > 3)
        assert result.to_list() == [4, 5]

    def test_unique(self):
        sa = SArray([1, 2, 2, 3, 3, 3])
        result = sa.unique()
        assert sorted(result.to_list()) == [1, 2, 3]

    def test_sort(self):
        sa = SArray([3, 1, 4, 1, 5])
        result = sa.sort()
        assert result.to_list() == [1, 1, 3, 4, 5]

    def test_logical_and_or(self):
        a = SArray([1, 0, 1])
        b = SArray([1, 1, 0])
        assert (a & b).to_list() == [1, 0, 0]
        assert (a | b).to_list() == [1, 1, 1]

    def test_repr(self):
        sa = SArray([1, 2, 3])
        r = repr(sa)
        assert "1" in r

    def test_iter(self):
        sa = SArray([10, 20, 30])
        values = list(sa)
        assert values == [10, 20, 30]

    def test_getitem(self):
        sa = SArray([10, 20, 30])
        assert sa[0] == 10
        assert sa[-1] == 30

    def test_contains_string(self):
        sa = SArray(["hello world", "foo bar", "hello"])
        result = sa.contains("hello")
        assert result.to_list() == [1, 0, 1]

    def test_clip(self):
        sa = SArray([1, 5, 10, 15, 20])
        result = sa.clip(5, 15)
        assert result.to_list() == [5, 5, 10, 15, 15]

    def test_bool_raises(self):
        sa = SArray([1, 2, 3])
        try:
            bool(sa)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_reverse_sub(self):
        sa = SArray([1, 2, 3])
        result = 10 - sa
        assert result.to_list() == [9, 8, 7]

    def test_mul_scalar(self):
        sa = SArray([2, 3, 4])
        result = sa * 3
        assert result.to_list() == [6, 9, 12]

    def test_radd(self):
        sa = SArray([1, 2, 3])
        result = 10 + sa
        assert result.to_list() == [11, 12, 13]


class TestSFrame:
    def test_read_save_roundtrip(self):
        """Test reading and saving an SFrame."""
        # This test requires the samples directory. Skip if not present.
        sf_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "samples", "business.sf"
        )
        if not os.path.exists(sf_path):
            return  # skip

        sf = SFrame.read(sf_path)
        assert sf.num_rows() > 0
        assert sf.num_columns() > 0
        assert len(sf.column_names()) == sf.num_columns()
        assert len(sf) == sf.num_rows()

        # Save and re-read
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.sf")
            sf.save(out_path)
            sf2 = SFrame.read(out_path)
            assert sf2.num_rows() == sf.num_rows()
            assert sf2.num_columns() == sf.num_columns()

    def test_from_columns(self):
        names = SArray(["Alice", "Bob", "Carol"])
        ages = SArray([30, 25, 35])
        sf = SFrame.from_columns({"name": names, "age": ages})
        assert sf.num_rows() == 3
        assert sf.num_columns() == 2
        assert set(sf.column_names()) == {"name", "age"}

    def test_getitem_column(self):
        names = SArray(["Alice", "Bob"])
        ages = SArray([30, 25])
        sf = SFrame.from_columns({"name": names, "age": ages})
        col = sf["name"]
        assert isinstance(col, SArray)
        assert col.to_list() == ["Alice", "Bob"]

    def test_getitem_mask(self):
        ages = SArray([30, 25, 35])
        names = SArray(["Alice", "Bob", "Carol"])
        sf = SFrame.from_columns({"name": names, "age": ages})
        mask = sf["age"] > 28
        filtered = sf[mask]
        assert filtered.num_rows() == 2

    def test_setitem(self):
        names = SArray(["Alice", "Bob"])
        ages = SArray([30, 25])
        sf = SFrame.from_columns({"name": names, "age": ages})
        sf["score"] = SArray([100, 200])
        assert "score" in sf
        assert sf.num_columns() == 3

    def test_contains(self):
        names = SArray(["Alice", "Bob"])
        sf = SFrame.from_columns({"name": names})
        assert "name" in sf
        assert "missing" not in sf

    def test_head_tail(self):
        sa = SArray(list(range(100)))
        sf = SFrame.from_columns({"val": sa})
        h = sf.head(5)
        assert h.num_rows() == 5
        t = sf.tail(5)
        assert t.num_rows() == 5

    def test_schema(self):
        names = SArray(["Alice", "Bob"])
        ages = SArray([30, 25])
        sf = SFrame.from_columns({"name": names, "age": ages})
        schema = sf.schema()
        schema_dict = dict(schema)
        assert schema_dict["name"] == "str"
        assert schema_dict["age"] == "int"

    def test_sort(self):
        ages = SArray([30, 25, 35])
        names = SArray(["Alice", "Bob", "Carol"])
        sf = SFrame.from_columns({"name": names, "age": ages})
        sorted_sf = sf.sort("age")
        col = sorted_sf["age"]
        assert col.to_list() == [25, 30, 35]

    def test_append(self):
        sf1 = SFrame.from_columns({"x": SArray([1, 2])})
        sf2 = SFrame.from_columns({"x": SArray([3, 4])})
        sf3 = sf1.append(sf2)
        assert sf3.num_rows() == 4

    def test_unique(self):
        sf = SFrame.from_columns({"x": SArray([1, 1, 2, 2, 3])})
        u = sf.unique()
        assert u.num_rows() == 3

    def test_filter_lambda(self):
        sf = SFrame.from_columns({"x": SArray([1, 2, 3, 4, 5])})
        result = sf.filter("x", lambda v: v > 3)
        assert result.num_rows() == 2

    def test_groupby(self):
        sf = SFrame.from_columns({
            "city": SArray(["A", "B", "A", "B", "A"]),
            "val": SArray([10, 20, 30, 40, 50]),
        })
        result = sf.groupby(["city"], {"total": aggregate.SUM("val")})
        assert result.num_rows() == 2
        # Check that totals are correct
        rows = result.iter_rows()
        totals = {r["city"]: r["total"] for r in rows}
        assert totals["A"] == 90
        assert totals["B"] == 60

    def test_join(self):
        left = SFrame.from_columns({
            "key": SArray([1, 2, 3]),
            "lval": SArray(["a", "b", "c"]),
        })
        right = SFrame.from_columns({
            "key": SArray([2, 3, 4]),
            "rval": SArray(["x", "y", "z"]),
        })
        result = left.join(right, "key", how="inner")
        assert result.num_rows() == 2

    def test_iter_rows(self):
        sf = SFrame.from_columns({
            "x": SArray([1, 2]),
            "y": SArray(["a", "b"]),
        })
        rows = sf.iter_rows()
        assert len(rows) == 2
        assert rows[0]["x"] == 1
        assert rows[0]["y"] == "a"

    def test_iter(self):
        sf = SFrame.from_columns({"x": SArray([10, 20])})
        rows = list(sf)
        assert len(rows) == 2
        assert rows[0]["x"] == 10

    def test_repr(self):
        sf = SFrame.from_columns({"x": SArray([1, 2, 3])})
        r = repr(sf)
        assert "x" in r

    def test_explain(self):
        sf = SFrame.from_columns({"x": SArray([1, 2, 3])})
        plan = sf.explain()
        assert isinstance(plan, str)

    def test_select_columns(self):
        sf = SFrame.from_columns({
            "a": SArray([1]),
            "b": SArray([2]),
            "c": SArray([3]),
        })
        sub = sf[["a", "c"]]
        assert sub.num_columns() == 2
        assert set(sub.column_names()) == {"a", "c"}

    def test_csv_roundtrip(self):
        sf = SFrame.from_columns({
            "x": SArray([1, 2, 3]),
            "y": SArray([4.0, 5.0, 6.0]),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            sf.to_csv(csv_path)
            sf2 = SFrame.from_csv(csv_path)
            assert sf2.num_rows() == 3


class TestStreamWriter:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stream.sf")
            writer = SFrameStreamWriter(path, ["x", "y"], ["int", "float"])
            writer.write_batch({"x": [1, 2, 3], "y": [1.5, 2.5, 3.5]})
            writer.write_batch({"x": [4, 5], "y": [4.5, 5.5]})
            writer.finish()

            sf = SFrame.read(path)
            assert sf.num_rows() == 5
            assert sf.num_columns() == 2

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ctx.sf")
            with SFrameStreamWriter(path, ["val"], ["int"]) as w:
                w.write_batch({"val": [10, 20, 30]})

            sf = SFrame.read(path)
            assert sf.num_rows() == 3


class TestAggregate:
    def test_sum(self):
        spec = aggregate.SUM("col")
        assert repr(spec) == "aggregate.SUM('col')"

    def test_count(self):
        spec = aggregate.COUNT()
        assert repr(spec) == "aggregate.COUNT('')"

    def test_all_agg_functions(self):
        """Verify all aggregate functions can be created."""
        aggregate.SUM("x")
        aggregate.MEAN("x")
        aggregate.MIN("x")
        aggregate.MAX("x")
        aggregate.COUNT("x")
        aggregate.COUNT()
        aggregate.VARIANCE("x")
        aggregate.STD("x")
        aggregate.COUNT_DISTINCT("x")
        aggregate.CONCAT("x")
        aggregate.SELECT_ONE("x")


class TestLoad:
    def test_load_function(self):
        sf_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "samples", "business.sf"
        )
        if not os.path.exists(sf_path):
            return  # skip
        sf = load(sf_path)
        assert sf.num_rows() > 0
