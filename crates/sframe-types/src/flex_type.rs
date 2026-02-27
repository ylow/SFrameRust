use std::sync::Arc;

use crate::error::{Result, SFrameError};

/// Type tag enum matching C++ flex_type_enum values for format compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FlexTypeEnum {
    Integer = 0,
    Float = 1,
    String = 2,
    Vector = 3,
    List = 4,
    Dict = 5,
    DateTime = 6,
    Undefined = 7,
}

impl TryFrom<u8> for FlexTypeEnum {
    type Error = SFrameError;
    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Integer),
            1 => Ok(Self::Float),
            2 => Ok(Self::String),
            3 => Ok(Self::Vector),
            4 => Ok(Self::List),
            5 => Ok(Self::Dict),
            6 => Ok(Self::DateTime),
            7 => Ok(Self::Undefined),
            _ => Err(SFrameError::Type(format!("Unknown type enum value: {}", value))),
        }
    }
}

impl std::fmt::Display for FlexTypeEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::String => write!(f, "string"),
            Self::Vector => write!(f, "vector"),
            Self::List => write!(f, "list"),
            Self::Dict => write!(f, "dict"),
            Self::DateTime => write!(f, "datetime"),
            Self::Undefined => write!(f, "undefined"),
        }
    }
}

/// Date/time with timezone and microsecond precision.
/// Matches C++ flex_date_time layout.
#[derive(Debug, Clone, PartialEq)]
pub struct FlexDateTime {
    pub posix_timestamp: i64,
    pub tz_offset_quarter_hours: i8,
    pub microsecond: u32,
}

/// The core value type, mirroring C++ flexible_type.
/// Uses Arc for shared ownership of heap-allocated data.
#[derive(Debug, Clone, PartialEq)]
pub enum FlexType {
    Integer(i64),
    Float(f64),
    String(Arc<str>),
    Vector(Arc<[f64]>),
    List(Arc<[FlexType]>),
    Dict(Arc<[(FlexType, FlexType)]>),
    DateTime(FlexDateTime),
    Undefined,
}

impl FlexType {
    /// Returns the type tag for this value.
    pub fn type_enum(&self) -> FlexTypeEnum {
        match self {
            FlexType::Integer(_) => FlexTypeEnum::Integer,
            FlexType::Float(_) => FlexTypeEnum::Float,
            FlexType::String(_) => FlexTypeEnum::String,
            FlexType::Vector(_) => FlexTypeEnum::Vector,
            FlexType::List(_) => FlexTypeEnum::List,
            FlexType::Dict(_) => FlexTypeEnum::Dict,
            FlexType::DateTime(_) => FlexTypeEnum::DateTime,
            FlexType::Undefined => FlexTypeEnum::Undefined,
        }
    }
}

impl std::fmt::Display for FlexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlexType::Integer(v) => write!(f, "{}", v),
            FlexType::Float(v) => write!(f, "{}", v),
            FlexType::String(v) => write!(f, "{}", v),
            FlexType::Vector(v) => {
                write!(f, "[")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", x)?;
                }
                write!(f, "]")
            }
            FlexType::List(v) => {
                write!(f, "[")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", x)?;
                }
                write!(f, "]")
            }
            FlexType::Dict(v) => {
                write!(f, "{{")?;
                for (i, (k, val)) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, val)?;
                }
                write!(f, "}}")
            }
            FlexType::DateTime(dt) => {
                write!(f, "DateTime({})", dt.posix_timestamp)
            }
            FlexType::Undefined => write!(f, "None"),
        }
    }
}

// === Arithmetic Operators ===
//
// Matching C++ flexible_type operator semantics:
// - int op int → int (except div → float)
// - float op float → float
// - int op float → float (promotion)
// - string + string → concat
// - vector op vector → element-wise (same length required)
// - vector op scalar → broadcast
// - Any op involving Undefined → Undefined

impl std::ops::Add for FlexType {
    type Output = FlexType;

    fn add(self, rhs: FlexType) -> FlexType {
        match (self, rhs) {
            (FlexType::Integer(a), FlexType::Integer(b)) => FlexType::Integer(a + b),
            (FlexType::Float(a), FlexType::Float(b)) => FlexType::Float(a + b),
            (FlexType::Integer(a), FlexType::Float(b)) => FlexType::Float(a as f64 + b),
            (FlexType::Float(a), FlexType::Integer(b)) => FlexType::Float(a + b as f64),
            (FlexType::String(a), FlexType::String(b)) => {
                let mut s = a.to_string();
                s.push_str(&b);
                FlexType::String(Arc::from(s.as_str()))
            }
            (FlexType::Vector(a), FlexType::Vector(b)) => {
                if a.len() == b.len() {
                    let v: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                    FlexType::Vector(Arc::from(v))
                } else {
                    FlexType::Undefined
                }
            }
            (FlexType::Undefined, _) | (_, FlexType::Undefined) => FlexType::Undefined,
            _ => FlexType::Undefined,
        }
    }
}

impl std::ops::Sub for FlexType {
    type Output = FlexType;

    fn sub(self, rhs: FlexType) -> FlexType {
        match (self, rhs) {
            (FlexType::Integer(a), FlexType::Integer(b)) => FlexType::Integer(a - b),
            (FlexType::Float(a), FlexType::Float(b)) => FlexType::Float(a - b),
            (FlexType::Integer(a), FlexType::Float(b)) => FlexType::Float(a as f64 - b),
            (FlexType::Float(a), FlexType::Integer(b)) => FlexType::Float(a - b as f64),
            (FlexType::Vector(a), FlexType::Vector(b)) => {
                if a.len() == b.len() {
                    let v: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
                    FlexType::Vector(Arc::from(v))
                } else {
                    FlexType::Undefined
                }
            }
            (FlexType::Undefined, _) | (_, FlexType::Undefined) => FlexType::Undefined,
            _ => FlexType::Undefined,
        }
    }
}

impl std::ops::Mul for FlexType {
    type Output = FlexType;

    fn mul(self, rhs: FlexType) -> FlexType {
        match (self, rhs) {
            (FlexType::Integer(a), FlexType::Integer(b)) => FlexType::Integer(a * b),
            (FlexType::Float(a), FlexType::Float(b)) => FlexType::Float(a * b),
            (FlexType::Integer(a), FlexType::Float(b)) => FlexType::Float(a as f64 * b),
            (FlexType::Float(a), FlexType::Integer(b)) => FlexType::Float(a * b as f64),
            (FlexType::Vector(a), FlexType::Vector(b)) => {
                if a.len() == b.len() {
                    let v: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
                    FlexType::Vector(Arc::from(v))
                } else {
                    FlexType::Undefined
                }
            }
            // vector * scalar
            (FlexType::Vector(a), FlexType::Float(s)) | (FlexType::Float(s), FlexType::Vector(a)) => {
                let v: Vec<f64> = a.iter().map(|x| x * s).collect();
                FlexType::Vector(Arc::from(v))
            }
            (FlexType::Vector(a), FlexType::Integer(s)) | (FlexType::Integer(s), FlexType::Vector(a)) => {
                let sf = s as f64;
                let v: Vec<f64> = a.iter().map(|x| x * sf).collect();
                FlexType::Vector(Arc::from(v))
            }
            (FlexType::Undefined, _) | (_, FlexType::Undefined) => FlexType::Undefined,
            _ => FlexType::Undefined,
        }
    }
}

impl std::ops::Div for FlexType {
    type Output = FlexType;

    /// Division always returns Float (like C++ SFrame).
    fn div(self, rhs: FlexType) -> FlexType {
        match (self, rhs) {
            (FlexType::Integer(a), FlexType::Integer(b)) => {
                if b == 0 {
                    FlexType::Undefined
                } else {
                    FlexType::Float(a as f64 / b as f64)
                }
            }
            (FlexType::Float(a), FlexType::Float(b)) => FlexType::Float(a / b),
            (FlexType::Integer(a), FlexType::Float(b)) => FlexType::Float(a as f64 / b),
            (FlexType::Float(a), FlexType::Integer(b)) => FlexType::Float(a / b as f64),
            (FlexType::Vector(a), FlexType::Vector(b)) => {
                if a.len() == b.len() {
                    let v: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
                    FlexType::Vector(Arc::from(v))
                } else {
                    FlexType::Undefined
                }
            }
            // vector / scalar
            (FlexType::Vector(a), FlexType::Float(s)) => {
                let v: Vec<f64> = a.iter().map(|x| x / s).collect();
                FlexType::Vector(Arc::from(v))
            }
            (FlexType::Vector(a), FlexType::Integer(s)) => {
                let sf = s as f64;
                let v: Vec<f64> = a.iter().map(|x| x / sf).collect();
                FlexType::Vector(Arc::from(v))
            }
            (FlexType::Undefined, _) | (_, FlexType::Undefined) => FlexType::Undefined,
            _ => FlexType::Undefined,
        }
    }
}

impl std::ops::Rem for FlexType {
    type Output = FlexType;

    fn rem(self, rhs: FlexType) -> FlexType {
        match (self, rhs) {
            (FlexType::Integer(a), FlexType::Integer(b)) => {
                if b == 0 {
                    FlexType::Undefined
                } else {
                    FlexType::Integer(a % b)
                }
            }
            (FlexType::Undefined, _) | (_, FlexType::Undefined) => FlexType::Undefined,
            _ => FlexType::Undefined,
        }
    }
}

impl std::ops::Neg for FlexType {
    type Output = FlexType;

    fn neg(self) -> FlexType {
        match self {
            FlexType::Integer(a) => FlexType::Integer(-a),
            FlexType::Float(a) => FlexType::Float(-a),
            FlexType::Vector(a) => {
                let v: Vec<f64> = a.iter().map(|x| -x).collect();
                FlexType::Vector(Arc::from(v))
            }
            FlexType::Undefined => FlexType::Undefined,
            _ => FlexType::Undefined,
        }
    }
}

// === Comparison & Ordering ===
//
// PartialOrd with cross-type numeric comparison.
// Undefined sorts last (greater than everything).
// Type ordering fallback: Integer < Float < String < Vector < List < Dict < DateTime < Undefined

impl PartialOrd for FlexType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            // Undefined sorts last
            (FlexType::Undefined, FlexType::Undefined) => Some(Ordering::Equal),
            (FlexType::Undefined, _) => Some(Ordering::Greater),
            (_, FlexType::Undefined) => Some(Ordering::Less),

            // Same-type comparisons
            (FlexType::Integer(a), FlexType::Integer(b)) => a.partial_cmp(b),
            (FlexType::Float(a), FlexType::Float(b)) => a.partial_cmp(b),
            (FlexType::String(a), FlexType::String(b)) => a.as_ref().partial_cmp(b.as_ref()),

            // Cross-type numeric
            (FlexType::Integer(a), FlexType::Float(b)) => (*a as f64).partial_cmp(b),
            (FlexType::Float(a), FlexType::Integer(b)) => a.partial_cmp(&(*b as f64)),

            // Vectors: lexicographic
            (FlexType::Vector(a), FlexType::Vector(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    match x.partial_cmp(y) {
                        Some(Ordering::Equal) => continue,
                        other => return other,
                    }
                }
                a.len().partial_cmp(&b.len())
            }

            // DateTime: by timestamp
            (FlexType::DateTime(a), FlexType::DateTime(b)) => {
                a.posix_timestamp.partial_cmp(&b.posix_timestamp)
            }

            // Different types: order by type rank
            (a, b) => {
                let rank_a = type_rank(a);
                let rank_b = type_rank(b);
                rank_a.partial_cmp(&rank_b)
            }
        }
    }
}

fn type_rank(v: &FlexType) -> u8 {
    match v {
        FlexType::Integer(_) => 0,
        FlexType::Float(_) => 1,
        FlexType::String(_) => 2,
        FlexType::Vector(_) => 3,
        FlexType::List(_) => 4,
        FlexType::Dict(_) => 5,
        FlexType::DateTime(_) => 6,
        FlexType::Undefined => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flex_type_enum_values() {
        assert_eq!(FlexTypeEnum::Integer as u8, 0);
        assert_eq!(FlexTypeEnum::Float as u8, 1);
        assert_eq!(FlexTypeEnum::String as u8, 2);
        assert_eq!(FlexTypeEnum::Vector as u8, 3);
        assert_eq!(FlexTypeEnum::List as u8, 4);
        assert_eq!(FlexTypeEnum::Dict as u8, 5);
        assert_eq!(FlexTypeEnum::DateTime as u8, 6);
        assert_eq!(FlexTypeEnum::Undefined as u8, 7);
    }

    #[test]
    fn test_flex_type_enum_from_u8() {
        assert_eq!(FlexTypeEnum::try_from(0u8).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(FlexTypeEnum::try_from(7u8).unwrap(), FlexTypeEnum::Undefined);
        assert!(FlexTypeEnum::try_from(8u8).is_err());
        assert!(FlexTypeEnum::try_from(255u8).is_err());
    }

    #[test]
    fn test_flex_type_type_tag() {
        assert_eq!(FlexType::Integer(42).type_enum(), FlexTypeEnum::Integer);
        assert_eq!(FlexType::Float(3.14).type_enum(), FlexTypeEnum::Float);
        assert_eq!(FlexType::String(Arc::from("hello")).type_enum(), FlexTypeEnum::String);
        assert_eq!(FlexType::Undefined.type_enum(), FlexTypeEnum::Undefined);
    }

    #[test]
    fn test_flex_type_display() {
        assert_eq!(format!("{}", FlexType::Integer(42)), "42");
        assert_eq!(format!("{}", FlexType::Float(3.14)), "3.14");
        assert_eq!(format!("{}", FlexType::String(Arc::from("hello"))), "hello");
        assert_eq!(format!("{}", FlexType::Undefined), "None");
    }

    // === Arithmetic operator tests ===

    #[test]
    fn test_add_integers() {
        assert_eq!(
            FlexType::Integer(3) + FlexType::Integer(4),
            FlexType::Integer(7)
        );
    }

    #[test]
    fn test_add_floats() {
        match FlexType::Float(1.5) + FlexType::Float(2.5) {
            FlexType::Float(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_add_int_float_promotion() {
        match FlexType::Integer(1) + FlexType::Float(2.5) {
            FlexType::Float(v) => assert!((v - 3.5).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_add_strings() {
        assert_eq!(
            FlexType::String(Arc::from("hello")) + FlexType::String(Arc::from(" world")),
            FlexType::String(Arc::from("hello world"))
        );
    }

    #[test]
    fn test_add_vectors() {
        let a = FlexType::Vector(Arc::from(vec![1.0, 2.0, 3.0]));
        let b = FlexType::Vector(Arc::from(vec![4.0, 5.0, 6.0]));
        match a + b {
            FlexType::Vector(v) => assert_eq!(v.as_ref(), &[5.0, 7.0, 9.0]),
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_sub_integers() {
        assert_eq!(
            FlexType::Integer(10) - FlexType::Integer(3),
            FlexType::Integer(7)
        );
    }

    #[test]
    fn test_mul_int_float() {
        match FlexType::Integer(3) * FlexType::Float(2.5) {
            FlexType::Float(v) => assert!((v - 7.5).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_mul_vector_scalar() {
        let v = FlexType::Vector(Arc::from(vec![1.0, 2.0, 3.0]));
        match v * FlexType::Float(2.0) {
            FlexType::Vector(r) => assert_eq!(r.as_ref(), &[2.0, 4.0, 6.0]),
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_div_always_float() {
        match FlexType::Integer(7) / FlexType::Integer(2) {
            FlexType::Float(v) => assert!((v - 3.5).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_div_by_zero() {
        assert_eq!(
            FlexType::Integer(1) / FlexType::Integer(0),
            FlexType::Undefined
        );
    }

    #[test]
    fn test_rem_integers() {
        assert_eq!(
            FlexType::Integer(7) % FlexType::Integer(3),
            FlexType::Integer(1)
        );
    }

    #[test]
    fn test_neg() {
        assert_eq!(-FlexType::Integer(5), FlexType::Integer(-5));
        match -FlexType::Float(3.14) {
            FlexType::Float(v) => assert!((v + 3.14).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_neg_vector() {
        match -FlexType::Vector(Arc::from(vec![1.0, -2.0, 3.0])) {
            FlexType::Vector(v) => assert_eq!(v.as_ref(), &[-1.0, 2.0, -3.0]),
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn test_undefined_propagates() {
        assert_eq!(
            FlexType::Integer(1) + FlexType::Undefined,
            FlexType::Undefined
        );
        assert_eq!(
            FlexType::Undefined - FlexType::Float(1.0),
            FlexType::Undefined
        );
    }

    // === Comparison / ordering tests ===

    #[test]
    fn test_ordering_integers() {
        assert!(FlexType::Integer(1) < FlexType::Integer(2));
        assert!(FlexType::Integer(3) > FlexType::Integer(2));
        assert!(FlexType::Integer(5) == FlexType::Integer(5));
    }

    #[test]
    fn test_ordering_floats() {
        assert!(FlexType::Float(1.0) < FlexType::Float(2.0));
    }

    #[test]
    fn test_ordering_cross_type_numeric() {
        assert!(FlexType::Integer(1) < FlexType::Float(1.5));
        assert!(FlexType::Float(0.5) < FlexType::Integer(1));
    }

    #[test]
    fn test_ordering_strings() {
        assert!(FlexType::String(Arc::from("apple")) < FlexType::String(Arc::from("banana")));
    }

    #[test]
    fn test_ordering_undefined_last() {
        assert!(FlexType::Integer(1) < FlexType::Undefined);
        assert!(FlexType::String(Arc::from("z")) < FlexType::Undefined);
        assert!(FlexType::Undefined == FlexType::Undefined);
    }

    #[test]
    fn test_ordering_different_types() {
        // Integer < Float < String < Vector
        assert!(FlexType::Integer(999) < FlexType::String(Arc::from("a")));
        assert!(FlexType::Float(999.0) < FlexType::String(Arc::from("a")));
    }
}
