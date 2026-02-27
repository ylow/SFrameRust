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
}
