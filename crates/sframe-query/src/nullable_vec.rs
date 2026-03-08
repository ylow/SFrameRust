//! A space-efficient nullable vector using a validity bitmap.
//!
//! `NullableVec<T>` stores values in a dense `Vec<MaybeUninit<T>>` with a
//! separate packed bitmap tracking which slots are initialized (Some) vs
//! null (None). This avoids the `Option<T>` discriminant overhead for types
//! without niche optimization (e.g. `i64`, `f64`).

use std::fmt;
use std::mem::MaybeUninit;

/// A space-efficient `Vec<Option<T>>` backed by a validity bitmap.
///
/// Values are stored in a dense `Vec<MaybeUninit<T>>`. A separate packed
/// `Vec<u64>` bitmap tracks which slots hold initialized values (Some)
/// versus null (None). This avoids the per-element discriminant overhead
/// that `Option<T>` adds for types without niche optimization.
pub struct NullableVec<T> {
    values: Vec<MaybeUninit<T>>,
    validity: Vec<u64>,
    len: usize,
}

// --- bitmap helpers ---

#[inline]
fn word_index(i: usize) -> usize {
    i / 64
}

#[inline]
fn bit_mask(i: usize) -> u64 {
    1u64 << (i % 64)
}

impl<T> NullableVec<T> {
    /// Create an empty `NullableVec`.
    pub fn new() -> Self {
        NullableVec {
            values: Vec::new(),
            validity: Vec::new(),
            len: 0,
        }
    }

    /// Create an empty `NullableVec` with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        NullableVec {
            values: Vec::with_capacity(cap),
            validity: Vec::with_capacity((cap + 63) / 64),
            len: 0,
        }
    }

    /// Number of elements (Some + None).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn is_valid(&self, i: usize) -> bool {
        self.validity[word_index(i)] & bit_mask(i) != 0
    }

    /// Push a value. `Some(v)` stores the value; `None` stores null.
    pub fn push(&mut self, value: Option<T>) {
        let idx = self.len;
        if word_index(idx) >= self.validity.len() {
            self.validity.push(0);
        }
        match value {
            Some(v) => {
                self.validity[word_index(idx)] |= bit_mask(idx);
                self.values.push(MaybeUninit::new(v));
            }
            None => {
                // Bit already 0 in the (possibly just-pushed) word.
                self.values.push(MaybeUninit::uninit());
            }
        }
        self.len += 1;
    }

    /// Get a reference to the value at `index`.
    ///
    /// Returns `Some(&T)` if the slot holds a value, `None` if null.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        assert!(index < self.len, "index {index} out of bounds (len {})", self.len);
        if self.is_valid(index) {
            // SAFETY: validity bit is set, so this slot was initialized via push(Some(..)).
            Some(unsafe { self.values[index].assume_init_ref() })
        } else {
            None
        }
    }

    /// Drop all elements and reset to empty.
    pub fn clear(&mut self) {
        // Drop initialized elements.
        for i in 0..self.len {
            if self.is_valid(i) {
                // SAFETY: validity bit is set → slot is initialized.
                unsafe { self.values[i].assume_init_drop(); }
            }
        }
        self.values.clear();
        self.validity.clear();
        self.len = 0;
    }

    /// Iterate over elements, yielding `Option<&T>`.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { vec: self, pos: 0 }
    }
}

impl<T: Clone> NullableVec<T> {
    /// Append all elements from `other`.
    pub fn extend_from_slice(&mut self, other: &NullableVec<T>) {
        self.values.reserve(other.len);
        for i in 0..other.len {
            self.push(other.get(i).cloned());
        }
    }
}

// --- Iterator ---

/// Iterator over `NullableVec<T>`, yielding `Option<&T>`.
pub struct Iter<'a, T> {
    vec: &'a NullableVec<T>,
    pos: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = Option<&'a T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.vec.len {
            return None;
        }
        let item = self.vec.get(self.pos);
        self.pos += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

// --- Trait impls ---

impl<T: Clone> Clone for NullableVec<T> {
    fn clone(&self) -> Self {
        let mut new_values = Vec::with_capacity(self.len);
        for i in 0..self.len {
            if self.is_valid(i) {
                // SAFETY: validity bit set → initialized.
                let val = unsafe { self.values[i].assume_init_ref() };
                new_values.push(MaybeUninit::new(val.clone()));
            } else {
                new_values.push(MaybeUninit::uninit());
            }
        }
        NullableVec {
            values: new_values,
            validity: self.validity.clone(),
            len: self.len,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for NullableVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter().map(|opt| {
                match opt {
                    Some(v) => format!("Some({v:?})"),
                    None => "None".to_string(),
                }
            }))
            .finish()
    }
}

impl<T> Drop for NullableVec<T> {
    fn drop(&mut self) {
        for i in 0..self.len {
            if self.is_valid(i) {
                // SAFETY: validity bit set → initialized.
                unsafe { self.values[i].assume_init_drop(); }
            }
        }
        // Vec<MaybeUninit<T>> drop just deallocates — MaybeUninit has no Drop.
        // We set len to 0 to prevent values.drop() from running drop on elements
        // (it won't anyway since MaybeUninit has no Drop, but this is defense-in-depth).
        self.values.clear();
    }
}

impl<T> FromIterator<Option<T>> for NullableVec<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut nv = NullableVec::with_capacity(lower);
        for item in iter {
            nv.push(item);
        }
        nv
    }
}

impl<T> Extend<Option<T>> for NullableVec<T> {
    fn extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T> From<Vec<Option<T>>> for NullableVec<T> {
    fn from(v: Vec<Option<T>>) -> Self {
        v.into_iter().collect()
    }
}

impl<T: PartialEq> PartialEq<Vec<Option<T>>> for NullableVec<T> {
    fn eq(&self, other: &Vec<Option<T>>) -> bool {
        if self.len != other.len() {
            return false;
        }
        for i in 0..self.len {
            match (self.get(i), &other[i]) {
                (Some(a), Some(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (None, None) => {}
                _ => return false,
            }
        }
        true
    }
}

impl<T: Clone> NullableVec<T> {
    /// Convert to `Vec<Option<T>>` (e.g. for Arrow interop).
    pub fn to_option_vec(&self) -> Vec<Option<T>> {
        self.iter().map(|opt| opt.cloned()).collect()
    }
}

impl<T> Default for NullableVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Construction ===

    #[test]
    fn new_is_empty() {
        let v: NullableVec<i64> = NullableVec::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn with_capacity_is_empty() {
        let v: NullableVec<i64> = NullableVec::with_capacity(100);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn from_vec_option() {
        let v = NullableVec::from(vec![Some(1i64), None, Some(3)]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Some(&1));
        assert_eq!(v.get(1), None);
        assert_eq!(v.get(2), Some(&3));
    }

    // === Push & Get ===

    #[test]
    fn push_some_and_get() {
        let mut v = NullableVec::new();
        v.push(Some(42i64));
        assert_eq!(v.len(), 1);
        assert_eq!(v.get(0), Some(&42));
    }

    #[test]
    fn push_none_and_get() {
        let mut v = NullableVec::new();
        v.push(None::<i64>);
        assert_eq!(v.len(), 1);
        assert_eq!(v.get(0), None);
    }

    #[test]
    fn push_mixed_sequence() {
        let mut v = NullableVec::new();
        for i in 0..200i64 {
            if i % 3 == 0 {
                v.push(None);
            } else {
                v.push(Some(i));
            }
        }
        assert_eq!(v.len(), 200);
        for i in 0..200i64 {
            if i % 3 == 0 {
                assert_eq!(v.get(i as usize), None, "index {i}");
            } else {
                assert_eq!(v.get(i as usize), Some(&i), "index {i}");
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_out_of_bounds_panics() {
        let v: NullableVec<i64> = NullableVec::new();
        v.get(0);
    }

    // === Iteration ===

    #[test]
    fn iter_empty() {
        let v: NullableVec<i64> = NullableVec::new();
        assert_eq!(v.iter().count(), 0);
    }

    #[test]
    fn iter_yields_option_ref() {
        let v = NullableVec::from(vec![Some(10i64), None, Some(30)]);
        let collected: Vec<Option<&i64>> = v.iter().collect();
        assert_eq!(collected, vec![Some(&10), None, Some(&30)]);
    }

    #[test]
    fn iter_many_elements() {
        let input: Vec<Option<i64>> = (0..200).map(|i| {
            if i % 5 == 0 { None } else { Some(i) }
        }).collect();
        let v = NullableVec::from(input.clone());
        let collected: Vec<Option<i64>> = v.iter().map(|o| o.copied()).collect();
        assert_eq!(collected, input);
    }

    // === Clone ===

    #[test]
    fn clone_preserves_values() {
        let v = NullableVec::from(vec![Some(1i64), None, Some(3)]);
        let v2 = v.clone();
        assert_eq!(v2.len(), 3);
        assert_eq!(v2.get(0), Some(&1));
        assert_eq!(v2.get(1), None);
        assert_eq!(v2.get(2), Some(&3));
    }

    #[test]
    fn clone_is_independent() {
        let mut v = NullableVec::from(vec![Some(1i64), Some(2)]);
        let v2 = v.clone();
        v.push(Some(3));
        assert_eq!(v.len(), 3);
        assert_eq!(v2.len(), 2);
    }

    // === extend_from_slice ===

    #[test]
    fn extend_from_slice_appends() {
        let mut a = NullableVec::from(vec![Some(1i64), None]);
        let b = NullableVec::from(vec![Some(3i64), Some(4)]);
        a.extend_from_slice(&b);
        assert_eq!(a.len(), 4);
        assert_eq!(a.get(0), Some(&1));
        assert_eq!(a.get(1), None);
        assert_eq!(a.get(2), Some(&3));
        assert_eq!(a.get(3), Some(&4));
    }

    #[test]
    fn extend_from_slice_empty_source() {
        let mut a = NullableVec::from(vec![Some(1i64)]);
        let b: NullableVec<i64> = NullableVec::new();
        a.extend_from_slice(&b);
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn extend_from_slice_into_empty() {
        let mut a: NullableVec<i64> = NullableVec::new();
        let b = NullableVec::from(vec![None, Some(2i64)]);
        a.extend_from_slice(&b);
        assert_eq!(a.len(), 2);
        assert_eq!(a.get(0), None);
        assert_eq!(a.get(1), Some(&2));
    }

    // === FromIterator ===

    #[test]
    fn collect_from_option_iter() {
        let v: NullableVec<i64> = vec![Some(1), None, Some(3)].into_iter().collect();
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Some(&1));
        assert_eq!(v.get(1), None);
        assert_eq!(v.get(2), Some(&3));
    }

    #[test]
    fn collect_from_map() {
        let source = vec![1i64, 2, 3];
        let v: NullableVec<i64> = source.into_iter().map(|x| {
            if x == 2 { None } else { Some(x * 10) }
        }).collect();
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Some(&10));
        assert_eq!(v.get(1), None);
        assert_eq!(v.get(2), Some(&30));
    }

    // === Extend trait ===

    #[test]
    fn extend_trait() {
        let mut v = NullableVec::from(vec![Some(1i64)]);
        v.extend(vec![None, Some(3)]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Some(&1));
        assert_eq!(v.get(1), None);
        assert_eq!(v.get(2), Some(&3));
    }

    // === Drop safety with heap types ===

    #[test]
    fn drop_with_arc_types() {
        use std::sync::Arc;
        let mut v: NullableVec<Arc<str>> = NullableVec::new();
        let s: Arc<str> = Arc::from("hello");
        v.push(Some(s.clone()));
        v.push(None);
        v.push(Some(Arc::from("world")));
        assert_eq!(Arc::strong_count(&s), 2); // s + inside v
        drop(v);
        assert_eq!(Arc::strong_count(&s), 1); // only s remains
    }

    #[test]
    fn clone_with_arc_types() {
        use std::sync::Arc;
        let s: Arc<str> = Arc::from("hello");
        let v = NullableVec::from(vec![Some(s.clone()), None]);
        assert_eq!(Arc::strong_count(&s), 2);
        let v2 = v.clone();
        assert_eq!(Arc::strong_count(&s), 3); // s + v + v2
        drop(v);
        assert_eq!(Arc::strong_count(&s), 2); // s + v2
        drop(v2);
        assert_eq!(Arc::strong_count(&s), 1);
    }

    // === Debug ===

    #[test]
    fn debug_format() {
        let v = NullableVec::from(vec![Some(1i64), None, Some(3)]);
        let s = format!("{:?}", v);
        assert!(s.contains("Some(1)"));
        assert!(s.contains("None"));
        assert!(s.contains("Some(3)"));
    }

    // === Edge cases ===

    #[test]
    fn all_none() {
        let v: NullableVec<i64> = vec![None, None, None].into_iter().collect();
        assert_eq!(v.len(), 3);
        for i in 0..3 {
            assert_eq!(v.get(i), None);
        }
    }

    #[test]
    fn all_some() {
        let v: NullableVec<i64> = vec![Some(1), Some(2), Some(3)].into_iter().collect();
        assert_eq!(v.len(), 3);
        for i in 0..3 {
            assert_eq!(v.get(i), Some(&(i as i64 + 1)));
        }
    }

    #[test]
    fn exactly_64_elements() {
        let v: NullableVec<i64> = (0..64).map(|i| {
            if i % 2 == 0 { Some(i) } else { None }
        }).collect();
        assert_eq!(v.len(), 64);
        for i in 0..64i64 {
            if i % 2 == 0 {
                assert_eq!(v.get(i as usize), Some(&i));
            } else {
                assert_eq!(v.get(i as usize), None);
            }
        }
    }

    #[test]
    fn across_word_boundary() {
        // Test elements at indices 62-65 (crossing the u64 word boundary)
        let mut v: NullableVec<i64> = NullableVec::new();
        for i in 0..66i64 {
            v.push(Some(i));
        }
        // Verify around the boundary
        assert_eq!(v.get(63), Some(&63));
        assert_eq!(v.get(64), Some(&64));
        assert_eq!(v.get(65), Some(&65));
    }

    #[test]
    fn clear_resets() {
        let mut v = NullableVec::from(vec![Some(1i64), None, Some(3)]);
        v.clear();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn clear_drops_values() {
        use std::sync::Arc;
        let s: Arc<str> = Arc::from("hello");
        let mut v = NullableVec::from(vec![Some(s.clone()), None]);
        assert_eq!(Arc::strong_count(&s), 2);
        v.clear();
        assert_eq!(Arc::strong_count(&s), 1);
        // Can still use after clear
        v.push(Some(Arc::from("world")));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn extend_from_slice_across_word_boundaries() {
        // a has 60 elements, b has 10 — extends across the 64-element boundary
        let a_data: Vec<Option<i64>> = (0..60).map(|i| Some(i)).collect();
        let b_data: Vec<Option<i64>> = (60..70).map(|i| Some(i)).collect();
        let mut a = NullableVec::from(a_data);
        let b = NullableVec::from(b_data);
        a.extend_from_slice(&b);
        assert_eq!(a.len(), 70);
        for i in 0..70i64 {
            assert_eq!(a.get(i as usize), Some(&i));
        }
    }

    // === PartialEq with Vec<Option<T>> ===

    #[test]
    fn eq_with_vec_option() {
        let nv = NullableVec::from(vec![Some(1i64), None, Some(3)]);
        let vo = vec![Some(1i64), None, Some(3)];
        assert_eq!(nv, vo);
    }

    #[test]
    fn ne_with_vec_option() {
        let nv = NullableVec::from(vec![Some(1i64), None, Some(3)]);
        let vo = vec![Some(1i64), Some(2), Some(3)];
        assert_ne!(nv, vo);
    }

    #[test]
    fn eq_with_vec_option_different_len() {
        let nv = NullableVec::from(vec![Some(1i64)]);
        let vo = vec![Some(1i64), None];
        assert_ne!(nv, vo);
    }

    // === to_option_vec ===

    #[test]
    fn to_option_vec_roundtrip() {
        let original = vec![Some(1i64), None, Some(3)];
        let nv = NullableVec::from(original.clone());
        assert_eq!(nv.to_option_vec(), original);
    }

    #[test]
    fn to_option_vec_empty() {
        let nv: NullableVec<i64> = NullableVec::new();
        let result: Vec<Option<i64>> = nv.to_option_vec();
        assert!(result.is_empty());
    }

    #[test]
    fn to_option_vec_all_none() {
        let nv = NullableVec::from(vec![None::<i64>, None, None]);
        assert_eq!(nv.to_option_vec(), vec![None, None, None]);
    }
}
