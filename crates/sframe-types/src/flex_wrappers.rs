//! Thin wrapper types for FlexType heap variants.
//!
//! These replace fat-pointer `Arc<str>`, `Arc<[f64]>`, etc. with thin-pointer
//! `ThinArc<(), T>` wrappers, reducing each from 16 bytes to 8 bytes on the stack.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use triomphe::ThinArc;

use crate::flex_type::FlexType;

// ---------------------------------------------------------------------------
// FlexString
// ---------------------------------------------------------------------------

/// Thin refcounted string. 8 bytes on the stack (single pointer).
///
/// Derefs to `&str`. Constructed from `&str` or `String`.
#[derive(Clone)]
pub struct FlexString(ThinArc<(), u8>);

impl FlexString {
    /// Create from a byte slice that is known to be valid UTF-8.
    ///
    /// # Safety
    /// The caller must ensure `bytes` is valid UTF-8.
    pub unsafe fn from_utf8_unchecked(bytes: &[u8]) -> Self {
        FlexString(ThinArc::from_header_and_slice((), bytes))
    }
}

impl Deref for FlexString {
    type Target = str;
    fn deref(&self) -> &str {
        // SAFETY: We only construct FlexString from valid UTF-8 sources.
        unsafe { std::str::from_utf8_unchecked(&self.0.slice) }
    }
}

impl AsRef<str> for FlexString {
    fn as_ref(&self) -> &str {
        self
    }
}

impl From<&str> for FlexString {
    fn from(s: &str) -> Self {
        FlexString(ThinArc::from_header_and_slice((), s.as_bytes()))
    }
}

impl From<String> for FlexString {
    fn from(s: String) -> Self {
        FlexString::from(s.as_str())
    }
}

impl PartialEq for FlexString {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for FlexString {}

impl Hash for FlexString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl PartialOrd for FlexString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl Ord for FlexString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl fmt::Debug for FlexString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl fmt::Display for FlexString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// FlexVec
// ---------------------------------------------------------------------------

/// Thin refcounted f64 vector. 8 bytes on the stack.
///
/// Derefs to `&[f64]`.
#[derive(Clone)]
pub struct FlexVec(ThinArc<(), f64>);

impl Deref for FlexVec {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.0.slice
    }
}

impl AsRef<[f64]> for FlexVec {
    fn as_ref(&self) -> &[f64] {
        self
    }
}

impl From<Vec<f64>> for FlexVec {
    fn from(v: Vec<f64>) -> Self {
        FlexVec(ThinArc::from_header_and_slice((), &v))
    }
}

impl From<&[f64]> for FlexVec {
    fn from(s: &[f64]) -> Self {
        FlexVec(ThinArc::from_header_and_slice((), s))
    }
}

impl PartialEq for FlexVec {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for FlexVec {}

impl Hash for FlexVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for x in self.iter() {
            x.to_bits().hash(state);
        }
    }
}

impl fmt::Debug for FlexVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl fmt::Display for FlexVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, x) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{x}")?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// FlexList
// ---------------------------------------------------------------------------

/// Thin refcounted list of FlexType values. 8 bytes on the stack.
///
/// Derefs to `&[FlexType]`.
#[derive(Clone)]
pub struct FlexList(ThinArc<(), FlexType>);

impl Deref for FlexList {
    type Target = [FlexType];
    fn deref(&self) -> &[FlexType] {
        &self.0.slice
    }
}

impl AsRef<[FlexType]> for FlexList {
    fn as_ref(&self) -> &[FlexType] {
        self
    }
}

impl From<Vec<FlexType>> for FlexList {
    fn from(v: Vec<FlexType>) -> Self {
        FlexList(ThinArc::from_header_and_iter((), v.into_iter()))
    }
}

impl PartialEq for FlexList {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for FlexList {}

impl Hash for FlexList {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl fmt::Debug for FlexList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// FlexDict
// ---------------------------------------------------------------------------

/// Thin refcounted dictionary (key-value pairs). 8 bytes on the stack.
///
/// Derefs to `&[(FlexType, FlexType)]`.
#[derive(Clone)]
pub struct FlexDict(ThinArc<(), (FlexType, FlexType)>);

impl Deref for FlexDict {
    type Target = [(FlexType, FlexType)];
    fn deref(&self) -> &[(FlexType, FlexType)] {
        &self.0.slice
    }
}

impl AsRef<[(FlexType, FlexType)]> for FlexDict {
    fn as_ref(&self) -> &[(FlexType, FlexType)] {
        self
    }
}

impl From<Vec<(FlexType, FlexType)>> for FlexDict {
    fn from(v: Vec<(FlexType, FlexType)>) -> Self {
        FlexDict(ThinArc::from_header_and_iter((), v.into_iter()))
    }
}

impl PartialEq for FlexDict {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for FlexDict {}

impl Hash for FlexDict {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl fmt::Debug for FlexDict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // --- Size assertions ---

    #[test]
    fn test_wrapper_sizes() {
        assert_eq!(std::mem::size_of::<FlexString>(), 8);
        assert_eq!(std::mem::size_of::<FlexVec>(), 8);
        assert_eq!(std::mem::size_of::<FlexList>(), 8);
        assert_eq!(std::mem::size_of::<FlexDict>(), 8);
    }

    // --- FlexString ---

    #[test]
    fn test_flexstring_from_str() {
        let s = FlexString::from("hello");
        assert_eq!(&*s, "hello");
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_flexstring_from_string() {
        let s = FlexString::from(String::from("world"));
        assert_eq!(&*s, "world");
    }

    #[test]
    fn test_flexstring_empty() {
        let s = FlexString::from("");
        assert_eq!(&*s, "");
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_flexstring_eq_hash() {
        let a = FlexString::from("hello");
        let b = FlexString::from("hello");
        let c = FlexString::from("world");
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn test_flexstring_clone() {
        let a = FlexString::from("shared");
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(&*a, &*b);
    }

    #[test]
    fn test_flexstring_display() {
        let s = FlexString::from("display me");
        assert_eq!(format!("{s}"), "display me");
    }

    #[test]
    fn test_flexstring_ord() {
        let a = FlexString::from("apple");
        let b = FlexString::from("banana");
        assert!(a < b);
    }

    #[test]
    fn test_flexstring_unicode() {
        let s = FlexString::from("héllo wörld \u{1f30d}");
        assert_eq!(&*s, "héllo wörld \u{1f30d}");
    }

    // --- FlexVec ---

    #[test]
    fn test_flexvec_from_vec() {
        let v = FlexVec::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(&*v, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_flexvec_from_slice() {
        let data = [4.0, 5.0];
        let v = FlexVec::from(data.as_slice());
        assert_eq!(&*v, &[4.0, 5.0]);
    }

    #[test]
    fn test_flexvec_empty() {
        let v = FlexVec::from(vec![]);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_flexvec_eq_bitwise() {
        // NaN == NaN via to_bits comparison
        let a = FlexVec::from(vec![f64::NAN]);
        let b = FlexVec::from(vec![f64::NAN]);
        assert_eq!(a, b);

        // -0.0 != 0.0 via to_bits
        let c = FlexVec::from(vec![0.0]);
        let d = FlexVec::from(vec![-0.0]);
        assert_ne!(c, d);
    }

    #[test]
    fn test_flexvec_hash() {
        let a = FlexVec::from(vec![1.0, 2.0]);
        let b = FlexVec::from(vec![1.0, 2.0]);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn test_flexvec_iter() {
        let v = FlexVec::from(vec![1.0, 2.0, 3.0]);
        let sum: f64 = v.iter().sum();
        assert!((sum - 6.0).abs() < f64::EPSILON);
    }

    // --- FlexList ---

    #[test]
    fn test_flexlist_from_vec() {
        let l = FlexList::from(vec![FlexType::Integer(1), FlexType::Float(2.0)]);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0], FlexType::Integer(1));
        assert_eq!(l[1], FlexType::Float(2.0));
    }

    #[test]
    fn test_flexlist_empty() {
        let l = FlexList::from(vec![]);
        assert!(l.is_empty());
    }

    #[test]
    fn test_flexlist_eq() {
        let a = FlexList::from(vec![FlexType::Integer(1)]);
        let b = FlexList::from(vec![FlexType::Integer(1)]);
        let c = FlexList::from(vec![FlexType::Integer(2)]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_flexlist_nested() {
        let inner = FlexList::from(vec![FlexType::Integer(42)]);
        let outer = FlexList::from(vec![FlexType::List(inner)]);
        assert_eq!(outer.len(), 1);
        match &outer[0] {
            FlexType::List(l) => {
                assert_eq!(l.len(), 1);
                assert_eq!(l[0], FlexType::Integer(42));
            }
            other => panic!("Expected List, got {other:?}"),
        }
    }

    // --- FlexDict ---

    #[test]
    fn test_flexdict_from_vec() {
        let d = FlexDict::from(vec![
            (FlexType::String(FlexString::from("key")), FlexType::Integer(1)),
        ]);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, FlexType::String(FlexString::from("key")));
        assert_eq!(d[0].1, FlexType::Integer(1));
    }

    #[test]
    fn test_flexdict_empty() {
        let d = FlexDict::from(vec![]);
        assert!(d.is_empty());
    }

    #[test]
    fn test_flexdict_eq() {
        let a = FlexDict::from(vec![
            (FlexType::String(FlexString::from("k")), FlexType::Integer(1)),
        ]);
        let b = FlexDict::from(vec![
            (FlexType::String(FlexString::from("k")), FlexType::Integer(1)),
        ]);
        assert_eq!(a, b);
    }
}
