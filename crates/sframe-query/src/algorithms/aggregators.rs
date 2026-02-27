//! Built-in aggregator implementations.
//!
//! Each aggregator implements the `Aggregator` trait for use in
//! groupby and reduce operations.

use std::any::Any;
use std::io::{Read, Write};

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::serialization::{
    read_f64, read_i64, read_u64, read_u8, write_f64, write_i64, write_u64, write_u8,
    read_flex_type, write_flex_type, read_string, write_string,
};

use crate::planner::Aggregator;

/// Count aggregator — counts non-undefined values.
#[derive(Clone)]
pub struct CountAggregator {
    count: u64,
}

impl CountAggregator {
    pub fn new() -> Self {
        CountAggregator { count: 0 }
    }
}

impl Default for CountAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for CountAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if !values.is_empty() && !matches!(values[0], FlexType::Undefined) {
            self.count += 1;
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<CountAggregator>() {
            self.count += o.count;
        }
    }

    fn finalize(&mut self) -> FlexType {
        FlexType::Integer(self.count as i64)
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Integer
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.count)
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.count = read_u64(reader)?;
        Ok(())
    }
}

/// Sum aggregator — sums numeric values.
#[derive(Clone)]
pub struct SumAggregator {
    int_sum: i64,
    float_sum: f64,
    is_float: bool,
    has_value: bool,
}

impl SumAggregator {
    pub fn new() -> Self {
        SumAggregator {
            int_sum: 0,
            float_sum: 0.0,
            is_float: false,
            has_value: false,
        }
    }
}

impl Default for SumAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for SumAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        match &values[0] {
            FlexType::Integer(i) => {
                self.int_sum += i;
                self.has_value = true;
            }
            FlexType::Float(f) => {
                self.float_sum += f;
                self.is_float = true;
                self.has_value = true;
            }
            _ => {}
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<SumAggregator>() {
            self.int_sum += o.int_sum;
            self.float_sum += o.float_sum;
            if o.is_float {
                self.is_float = true;
            }
            if o.has_value {
                self.has_value = true;
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        if !self.has_value {
            return FlexType::Undefined;
        }
        if self.is_float {
            FlexType::Float(self.float_sum + self.int_sum as f64)
        } else {
            FlexType::Integer(self.int_sum)
        }
    }

    fn output_type(&self, input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        if !input_types.is_empty() && input_types[0] == FlexTypeEnum::Float {
            FlexTypeEnum::Float
        } else {
            FlexTypeEnum::Integer
        }
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_i64(writer, self.int_sum)?;
        write_f64(writer, self.float_sum)?;
        write_u8(writer, self.is_float as u8)?;
        write_u8(writer, self.has_value as u8)?;
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.int_sum = read_i64(reader)?;
        self.float_sum = read_f64(reader)?;
        self.is_float = read_u8(reader)? != 0;
        self.has_value = read_u8(reader)? != 0;
        Ok(())
    }
}

/// Mean aggregator — computes arithmetic mean.
#[derive(Clone)]
pub struct MeanAggregator {
    sum: f64,
    count: u64,
}

impl MeanAggregator {
    pub fn new() -> Self {
        MeanAggregator { sum: 0.0, count: 0 }
    }
}

impl Default for MeanAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for MeanAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        match &values[0] {
            FlexType::Integer(i) => {
                self.sum += *i as f64;
                self.count += 1;
            }
            FlexType::Float(f) => {
                self.sum += f;
                self.count += 1;
            }
            _ => {}
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<MeanAggregator>() {
            self.sum += o.sum;
            self.count += o.count;
        }
    }

    fn finalize(&mut self) -> FlexType {
        if self.count == 0 {
            FlexType::Undefined
        } else {
            FlexType::Float(self.sum / self.count as f64)
        }
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Float
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_f64(writer, self.sum)?;
        write_u64(writer, self.count)?;
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.sum = read_f64(reader)?;
        self.count = read_u64(reader)?;
        Ok(())
    }
}

/// Min aggregator — finds the minimum value.
#[derive(Clone)]
pub struct MinAggregator {
    min_int: Option<i64>,
    min_float: Option<f64>,
}

impl MinAggregator {
    pub fn new() -> Self {
        MinAggregator {
            min_int: None,
            min_float: None,
        }
    }
}

impl Default for MinAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for MinAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        match &values[0] {
            FlexType::Integer(i) => {
                self.min_int = Some(self.min_int.map_or(*i, |m| m.min(*i)));
            }
            FlexType::Float(f) => {
                self.min_float = Some(self.min_float.map_or(*f, |m| m.min(*f)));
            }
            _ => {}
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<MinAggregator>() {
            if let Some(oi) = o.min_int {
                self.min_int = Some(self.min_int.map_or(oi, |m| m.min(oi)));
            }
            if let Some(of) = o.min_float {
                self.min_float = Some(self.min_float.map_or(of, |m| m.min(of)));
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        match (self.min_int, self.min_float) {
            (Some(i), None) => FlexType::Integer(i),
            (None, Some(f)) => FlexType::Float(f),
            (Some(i), Some(f)) => {
                if (i as f64) <= f {
                    FlexType::Integer(i)
                } else {
                    FlexType::Float(f)
                }
            }
            (None, None) => FlexType::Undefined,
        }
    }

    fn output_type(&self, input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        if !input_types.is_empty() {
            input_types[0]
        } else {
            FlexTypeEnum::Undefined
        }
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u8(writer, self.min_int.is_some() as u8)?;
        write_i64(writer, self.min_int.unwrap_or(0))?;
        write_u8(writer, self.min_float.is_some() as u8)?;
        write_f64(writer, self.min_float.unwrap_or(0.0))?;
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let has_int = read_u8(reader)? != 0;
        let int_val = read_i64(reader)?;
        self.min_int = if has_int { Some(int_val) } else { None };
        let has_float = read_u8(reader)? != 0;
        let float_val = read_f64(reader)?;
        self.min_float = if has_float { Some(float_val) } else { None };
        Ok(())
    }
}

/// Max aggregator — finds the maximum value.
#[derive(Clone)]
pub struct MaxAggregator {
    max_int: Option<i64>,
    max_float: Option<f64>,
}

impl MaxAggregator {
    pub fn new() -> Self {
        MaxAggregator {
            max_int: None,
            max_float: None,
        }
    }
}

impl Default for MaxAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for MaxAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        match &values[0] {
            FlexType::Integer(i) => {
                self.max_int = Some(self.max_int.map_or(*i, |m| m.max(*i)));
            }
            FlexType::Float(f) => {
                self.max_float = Some(self.max_float.map_or(*f, |m| m.max(*f)));
            }
            _ => {}
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<MaxAggregator>() {
            if let Some(oi) = o.max_int {
                self.max_int = Some(self.max_int.map_or(oi, |m| m.max(oi)));
            }
            if let Some(of) = o.max_float {
                self.max_float = Some(self.max_float.map_or(of, |m| m.max(of)));
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        match (self.max_int, self.max_float) {
            (Some(i), None) => FlexType::Integer(i),
            (None, Some(f)) => FlexType::Float(f),
            (Some(i), Some(f)) => {
                if (i as f64) >= f {
                    FlexType::Integer(i)
                } else {
                    FlexType::Float(f)
                }
            }
            (None, None) => FlexType::Undefined,
        }
    }

    fn output_type(&self, input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        if !input_types.is_empty() {
            input_types[0]
        } else {
            FlexTypeEnum::Undefined
        }
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u8(writer, self.max_int.is_some() as u8)?;
        write_i64(writer, self.max_int.unwrap_or(0))?;
        write_u8(writer, self.max_float.is_some() as u8)?;
        write_f64(writer, self.max_float.unwrap_or(0.0))?;
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let has_int = read_u8(reader)? != 0;
        let int_val = read_i64(reader)?;
        self.max_int = if has_int { Some(int_val) } else { None };
        let has_float = read_u8(reader)? != 0;
        let float_val = read_f64(reader)?;
        self.max_float = if has_float { Some(float_val) } else { None };
        Ok(())
    }
}

/// Variance aggregator — computes sample variance using Welford's algorithm.
#[derive(Clone)]
pub struct VarianceAggregator {
    count: u64,
    mean: f64,
    m2: f64,
    ddof: u64, // 0 for population, 1 for sample
}

impl VarianceAggregator {
    pub fn new(ddof: u64) -> Self {
        VarianceAggregator {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            ddof,
        }
    }

    /// Sample variance (ddof=1).
    pub fn sample() -> Self {
        Self::new(1)
    }

    /// Population variance (ddof=0).
    pub fn population() -> Self {
        Self::new(0)
    }
}

impl Aggregator for VarianceAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        let x = match &values[0] {
            FlexType::Integer(i) => *i as f64,
            FlexType::Float(f) => *f,
            _ => return,
        };
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<VarianceAggregator>() {
            if o.count == 0 {
                return;
            }
            if self.count == 0 {
                self.count = o.count;
                self.mean = o.mean;
                self.m2 = o.m2;
                return;
            }
            let total = self.count + o.count;
            let delta = o.mean - self.mean;
            self.m2 += o.m2
                + delta * delta * (self.count as f64 * o.count as f64) / total as f64;
            self.mean = (self.mean * self.count as f64 + o.mean * o.count as f64) / total as f64;
            self.count = total;
        }
    }

    fn finalize(&mut self) -> FlexType {
        if self.count <= self.ddof {
            return FlexType::Undefined;
        }
        FlexType::Float(self.m2 / (self.count - self.ddof) as f64)
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Float
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.count)?;
        write_f64(writer, self.mean)?;
        write_f64(writer, self.m2)?;
        write_u64(writer, self.ddof)?;
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.count = read_u64(reader)?;
        self.mean = read_f64(reader)?;
        self.m2 = read_f64(reader)?;
        self.ddof = read_u64(reader)?;
        Ok(())
    }
}

/// StdDev aggregator — standard deviation (sqrt of variance).
#[derive(Clone)]
pub struct StdDevAggregator {
    var_agg: VarianceAggregator,
}

impl StdDevAggregator {
    pub fn sample() -> Self {
        StdDevAggregator {
            var_agg: VarianceAggregator::sample(),
        }
    }

    pub fn population() -> Self {
        StdDevAggregator {
            var_agg: VarianceAggregator::population(),
        }
    }
}

impl Aggregator for StdDevAggregator {
    fn add(&mut self, values: &[FlexType]) {
        self.var_agg.add(values);
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<StdDevAggregator>() {
            self.var_agg.merge(&o.var_agg);
        }
    }

    fn finalize(&mut self) -> FlexType {
        match self.var_agg.finalize() {
            FlexType::Float(v) => FlexType::Float(v.sqrt()),
            other => other,
        }
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Float
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        self.var_agg.save(writer)
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.var_agg.load(reader)
    }
}

/// Concat aggregator — collects values into a List.
#[derive(Clone)]
pub struct ConcatAggregator {
    values: Vec<FlexType>,
}

impl ConcatAggregator {
    pub fn new() -> Self {
        ConcatAggregator { values: Vec::new() }
    }
}

impl Default for ConcatAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for ConcatAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if !values.is_empty() {
            self.values.push(values[0].clone());
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<ConcatAggregator>() {
            self.values.extend_from_slice(&o.values);
        }
    }

    fn finalize(&mut self) -> FlexType {
        FlexType::List(self.values.clone().into())
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::List
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.values.len() as u64)?;
        for v in &self.values {
            write_flex_type(writer, v)?;
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let len = read_u64(reader)? as usize;
        self.values.clear();
        self.values.reserve(len);
        for _ in 0..len {
            self.values.push(read_flex_type(reader)?);
        }
        Ok(())
    }
}

/// Count distinct (unique) values aggregator.
#[derive(Clone)]
pub struct CountDistinctAggregator {
    /// Store string representations for hashing since FlexType doesn't impl Hash.
    seen: std::collections::HashSet<String>,
}

impl CountDistinctAggregator {
    pub fn new() -> Self {
        CountDistinctAggregator {
            seen: std::collections::HashSet::new(),
        }
    }
}

impl Default for CountDistinctAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for CountDistinctAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        if !matches!(values[0], FlexType::Undefined) {
            self.seen.insert(format!("{}", values[0]));
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<CountDistinctAggregator>() {
            self.seen.extend(o.seen.iter().cloned());
        }
    }

    fn finalize(&mut self) -> FlexType {
        FlexType::Integer(self.seen.len() as i64)
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Integer
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.seen.len() as u64)?;
        for s in &self.seen {
            write_string(writer, s)?;
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let len = read_u64(reader)? as usize;
        self.seen.clear();
        self.seen.reserve(len);
        for _ in 0..len {
            self.seen.insert(read_string(reader)?);
        }
        Ok(())
    }
}

/// Count non-Undefined values aggregator.
#[derive(Clone)]
pub struct NonNullCountAggregator {
    count: u64,
}

impl NonNullCountAggregator {
    pub fn new() -> Self {
        NonNullCountAggregator { count: 0 }
    }
}

impl Default for NonNullCountAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for NonNullCountAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if !values.is_empty() && !matches!(values[0], FlexType::Undefined) {
            self.count += 1;
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<NonNullCountAggregator>() {
            self.count += o.count;
        }
    }

    fn finalize(&mut self) -> FlexType {
        FlexType::Integer(self.count as i64)
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Integer
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.count)
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.count = read_u64(reader)?;
        Ok(())
    }
}

/// Select one (first non-Undefined) value aggregator.
#[derive(Clone)]
pub struct SelectOneAggregator {
    value: Option<FlexType>,
}

impl SelectOneAggregator {
    pub fn new() -> Self {
        SelectOneAggregator { value: None }
    }
}

impl Default for SelectOneAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for SelectOneAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if self.value.is_none() && !values.is_empty() && !matches!(values[0], FlexType::Undefined) {
            self.value = Some(values[0].clone());
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if self.value.is_none() {
            if let Some(o) = other.as_any().downcast_ref::<SelectOneAggregator>() {
                self.value = o.value.clone();
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        self.value.take().unwrap_or(FlexType::Undefined)
    }

    fn output_type(&self, input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        if !input_types.is_empty() {
            input_types[0]
        } else {
            FlexTypeEnum::Undefined
        }
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u8(writer, self.value.is_some() as u8)?;
        if let Some(v) = &self.value {
            write_flex_type(writer, v)?;
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let has_value = read_u8(reader)? != 0;
        self.value = if has_value {
            Some(read_flex_type(reader)?)
        } else {
            None
        };
        Ok(())
    }
}

/// Quantile aggregator — collects all values, sorts, and picks the percentile.
#[derive(Clone)]
pub struct QuantileAggregator {
    values: Vec<f64>,
    quantile: f64,
}

impl QuantileAggregator {
    pub fn new(quantile: f64) -> Self {
        QuantileAggregator {
            values: Vec::new(),
            quantile: quantile.clamp(0.0, 1.0),
        }
    }

    pub fn median() -> Self {
        Self::new(0.5)
    }
}

impl Aggregator for QuantileAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if values.is_empty() {
            return;
        }
        match &values[0] {
            FlexType::Integer(i) => self.values.push(*i as f64),
            FlexType::Float(f) => self.values.push(*f),
            _ => {}
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<QuantileAggregator>() {
            self.values.extend_from_slice(&o.values);
        }
    }

    fn finalize(&mut self) -> FlexType {
        if self.values.is_empty() {
            return FlexType::Undefined;
        }
        self.values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = self.values.len();
        let idx = ((n - 1) as f64 * self.quantile).round() as usize;
        let idx = idx.min(n - 1);
        FlexType::Float(self.values[idx])
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Float
    }

    fn box_clone(&self) -> Box<dyn Aggregator> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_f64(writer, self.quantile)?;
        write_u64(writer, self.values.len() as u64)?;
        for &v in &self.values {
            write_f64(writer, v)?;
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.quantile = read_f64(reader)?;
        let len = read_u64(reader)? as usize;
        self.values.clear();
        self.values.reserve(len);
        for _ in 0..len {
            self.values.push(read_f64(reader)?);
        }
        Ok(())
    }
}

/// Frequency count aggregator — returns a Dict mapping value → count.
#[derive(Clone)]
pub struct FrequencyCountAggregator {
    counts: std::collections::HashMap<String, u64>,
}

impl FrequencyCountAggregator {
    pub fn new() -> Self {
        FrequencyCountAggregator { counts: std::collections::HashMap::new() }
    }
}

impl Aggregator for FrequencyCountAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if let Some(v) = values.first() {
            if !matches!(v, FlexType::Undefined) {
                let key = format!("{}", v);
                *self.counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<Self>() {
            for (k, v) in &o.counts {
                *self.counts.entry(k.clone()).or_insert(0) += v;
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        let entries: Vec<(FlexType, FlexType)> = self.counts
            .iter()
            .map(|(k, &v)| (FlexType::String(k.clone().into()), FlexType::Integer(v as i64)))
            .collect();
        FlexType::Dict(std::sync::Arc::from(entries))
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Dict
    }

    fn box_clone(&self) -> Box<dyn Aggregator> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.counts.len() as u64)?;
        for (k, &v) in &self.counts {
            write_string(writer, k)?;
            write_u64(writer, v)?;
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let len = read_u64(reader)? as usize;
        self.counts.clear();
        self.counts.reserve(len);
        for _ in 0..len {
            let k = read_string(reader)?;
            let v = read_u64(reader)?;
            self.counts.insert(k, v);
        }
        Ok(())
    }
}

/// Collect values into a List.
#[derive(Clone)]
pub struct ZipListAggregator {
    values: Vec<FlexType>,
}

impl ZipListAggregator {
    pub fn new() -> Self {
        ZipListAggregator { values: Vec::new() }
    }
}

impl Aggregator for ZipListAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if let Some(v) = values.first() {
            self.values.push(v.clone());
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<Self>() {
            self.values.extend(o.values.iter().cloned());
        }
    }

    fn finalize(&mut self) -> FlexType {
        FlexType::List(std::sync::Arc::from(self.values.clone()))
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::List
    }

    fn box_clone(&self) -> Box<dyn Aggregator> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u64(writer, self.values.len() as u64)?;
        for v in &self.values {
            write_flex_type(writer, v)?;
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let len = read_u64(reader)? as usize;
        self.values.clear();
        self.values.reserve(len);
        for _ in 0..len {
            self.values.push(read_flex_type(reader)?);
        }
        Ok(())
    }
}

/// Element-wise sum of Vector columns.
#[derive(Clone)]
pub struct VectorSumAggregator {
    sum: Option<Vec<f64>>,
}

impl VectorSumAggregator {
    pub fn new() -> Self {
        VectorSumAggregator { sum: None }
    }
}

impl Aggregator for VectorSumAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if let Some(FlexType::Vector(v)) = values.first() {
            match &mut self.sum {
                None => self.sum = Some(v.to_vec()),
                Some(s) => {
                    for (i, &val) in v.iter().enumerate() {
                        if i < s.len() {
                            s[i] += val;
                        }
                    }
                }
            }
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<Self>() {
            if let Some(other_sum) = &o.sum {
                match &mut self.sum {
                    None => self.sum = Some(other_sum.clone()),
                    Some(s) => {
                        for (i, &val) in other_sum.iter().enumerate() {
                            if i < s.len() { s[i] += val; }
                        }
                    }
                }
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        match &self.sum {
            Some(s) => FlexType::Vector(std::sync::Arc::from(s.clone())),
            None => FlexType::Undefined,
        }
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Vector
    }

    fn box_clone(&self) -> Box<dyn Aggregator> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u8(writer, self.sum.is_some() as u8)?;
        if let Some(s) = &self.sum {
            write_u64(writer, s.len() as u64)?;
            for &v in s {
                write_f64(writer, v)?;
            }
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let has_value = read_u8(reader)? != 0;
        self.sum = if has_value {
            let len = read_u64(reader)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_f64(reader)?);
            }
            Some(v)
        } else {
            None
        };
        Ok(())
    }
}

/// Element-wise average of Vector columns.
#[derive(Clone)]
pub struct VectorAvgAggregator {
    sum: Option<Vec<f64>>,
    count: u64,
}

impl VectorAvgAggregator {
    pub fn new() -> Self {
        VectorAvgAggregator { sum: None, count: 0 }
    }
}

impl Aggregator for VectorAvgAggregator {
    fn add(&mut self, values: &[FlexType]) {
        if let Some(FlexType::Vector(v)) = values.first() {
            self.count += 1;
            match &mut self.sum {
                None => self.sum = Some(v.to_vec()),
                Some(s) => {
                    for (i, &val) in v.iter().enumerate() {
                        if i < s.len() { s[i] += val; }
                    }
                }
            }
        }
    }

    fn merge(&mut self, other: &dyn Aggregator) {
        if let Some(o) = other.as_any().downcast_ref::<Self>() {
            self.count += o.count;
            if let Some(other_sum) = &o.sum {
                match &mut self.sum {
                    None => self.sum = Some(other_sum.clone()),
                    Some(s) => {
                        for (i, &val) in other_sum.iter().enumerate() {
                            if i < s.len() { s[i] += val; }
                        }
                    }
                }
            }
        }
    }

    fn finalize(&mut self) -> FlexType {
        match &self.sum {
            Some(s) if self.count > 0 => {
                let avg: Vec<f64> = s.iter().map(|v| v / self.count as f64).collect();
                FlexType::Vector(std::sync::Arc::from(avg))
            }
            _ => FlexType::Undefined,
        }
    }

    fn output_type(&self, _input_types: &[FlexTypeEnum]) -> FlexTypeEnum {
        FlexTypeEnum::Vector
    }

    fn box_clone(&self) -> Box<dyn Aggregator> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }

    fn save(&self, writer: &mut dyn Write) -> Result<()> {
        write_u8(writer, self.sum.is_some() as u8)?;
        if let Some(s) = &self.sum {
            write_u64(writer, s.len() as u64)?;
            for &v in s {
                write_f64(writer, v)?;
            }
        }
        write_u64(writer, self.count)?;
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> Result<()> {
        let has_value = read_u8(reader)? != 0;
        self.sum = if has_value {
            let len = read_u64(reader)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_f64(reader)?);
            }
            Some(v)
        } else {
            None
        };
        self.count = read_u64(reader)?;
        Ok(())
    }
}

/// Specification for a groupby aggregation.
#[derive(Clone)]
pub struct AggSpec {
    /// Which column to aggregate on.
    pub column: usize,
    /// The aggregator to use.
    pub aggregator: Box<dyn Aggregator>,
    /// Output column name.
    pub output_name: String,
}

impl AggSpec {
    pub fn new(column: usize, aggregator: Box<dyn Aggregator>, output_name: &str) -> Self {
        AggSpec {
            column,
            aggregator,
            output_name: output_name.to_string(),
        }
    }

    pub fn count(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(CountAggregator::new()), output_name)
    }

    pub fn sum(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(SumAggregator::new()), output_name)
    }

    pub fn mean(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(MeanAggregator::new()), output_name)
    }

    pub fn min(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(MinAggregator::new()), output_name)
    }

    pub fn max(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(MaxAggregator::new()), output_name)
    }

    pub fn variance(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(VarianceAggregator::sample()), output_name)
    }

    pub fn stddev(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(StdDevAggregator::sample()), output_name)
    }

    pub fn concat(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(ConcatAggregator::new()), output_name)
    }

    pub fn count_distinct(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(CountDistinctAggregator::new()), output_name)
    }

    pub fn quantile(column: usize, quantile: f64, output_name: &str) -> Self {
        Self::new(column, Box::new(QuantileAggregator::new(quantile)), output_name)
    }

    pub fn median(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(QuantileAggregator::median()), output_name)
    }

    pub fn select_one(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(SelectOneAggregator::new()), output_name)
    }

    pub fn frequency_count(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(FrequencyCountAggregator::new()), output_name)
    }

    pub fn zip_list(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(ZipListAggregator::new()), output_name)
    }

    pub fn vector_sum(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(VectorSumAggregator::new()), output_name)
    }

    pub fn vector_avg(column: usize, output_name: &str) -> Self {
        Self::new(column, Box::new(VectorAvgAggregator::new()), output_name)
    }
}

impl Clone for Box<dyn Aggregator> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count() {
        let mut agg = CountAggregator::new();
        agg.add(&[FlexType::Integer(1)]);
        agg.add(&[FlexType::Integer(2)]);
        agg.add(&[FlexType::Undefined]);
        agg.add(&[FlexType::Integer(3)]);
        assert_eq!(agg.finalize(), FlexType::Integer(3));
    }

    #[test]
    fn test_sum_int() {
        let mut agg = SumAggregator::new();
        agg.add(&[FlexType::Integer(10)]);
        agg.add(&[FlexType::Integer(20)]);
        agg.add(&[FlexType::Integer(30)]);
        assert_eq!(agg.finalize(), FlexType::Integer(60));
    }

    #[test]
    fn test_sum_float() {
        let mut agg = SumAggregator::new();
        agg.add(&[FlexType::Float(1.5)]);
        agg.add(&[FlexType::Float(2.5)]);
        match agg.finalize() {
            FlexType::Float(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_mean() {
        let mut agg = MeanAggregator::new();
        agg.add(&[FlexType::Float(2.0)]);
        agg.add(&[FlexType::Float(4.0)]);
        agg.add(&[FlexType::Float(6.0)]);
        match agg.finalize() {
            FlexType::Float(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_min_max() {
        let mut min_agg = MinAggregator::new();
        let mut max_agg = MaxAggregator::new();
        for &v in &[5, 2, 8, 1, 9] {
            min_agg.add(&[FlexType::Integer(v)]);
            max_agg.add(&[FlexType::Integer(v)]);
        }
        assert_eq!(min_agg.finalize(), FlexType::Integer(1));
        assert_eq!(max_agg.finalize(), FlexType::Integer(9));
    }

    #[test]
    fn test_variance() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Population variance = 4.0
        let mut agg = VarianceAggregator::population();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            agg.add(&[FlexType::Float(v)]);
        }
        match agg.finalize() {
            FlexType::Float(v) => assert!((v - 4.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_stddev() {
        let mut agg = StdDevAggregator::population();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            agg.add(&[FlexType::Float(v)]);
        }
        match agg.finalize() {
            FlexType::Float(v) => assert!((v - 2.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_merge() {
        let mut agg1 = SumAggregator::new();
        let mut agg2 = SumAggregator::new();
        agg1.add(&[FlexType::Integer(10)]);
        agg1.add(&[FlexType::Integer(20)]);
        agg2.add(&[FlexType::Integer(30)]);
        agg2.add(&[FlexType::Integer(40)]);
        agg1.merge(&agg2);
        assert_eq!(agg1.finalize(), FlexType::Integer(100));
    }

    #[test]
    fn test_concat() {
        let mut agg = ConcatAggregator::new();
        agg.add(&[FlexType::String("a".into())]);
        agg.add(&[FlexType::String("b".into())]);
        agg.add(&[FlexType::String("c".into())]);
        match agg.finalize() {
            FlexType::List(l) => {
                assert_eq!(l.len(), 3);
                assert_eq!(l[0], FlexType::String("a".into()));
                assert_eq!(l[1], FlexType::String("b".into()));
                assert_eq!(l[2], FlexType::String("c".into()));
            }
            other => panic!("Expected List, got {:?}", other),
        }
    }

    #[test]
    fn test_count_distinct() {
        let mut agg = CountDistinctAggregator::new();
        agg.add(&[FlexType::Integer(1)]);
        agg.add(&[FlexType::Integer(2)]);
        agg.add(&[FlexType::Integer(1)]);
        agg.add(&[FlexType::Integer(3)]);
        agg.add(&[FlexType::Undefined]);
        assert_eq!(agg.finalize(), FlexType::Integer(3));
    }

    #[test]
    fn test_non_null_count() {
        let mut agg = NonNullCountAggregator::new();
        agg.add(&[FlexType::Integer(1)]);
        agg.add(&[FlexType::Undefined]);
        agg.add(&[FlexType::Integer(3)]);
        agg.add(&[FlexType::Undefined]);
        assert_eq!(agg.finalize(), FlexType::Integer(2));
    }

    #[test]
    fn test_select_one() {
        let mut agg = SelectOneAggregator::new();
        agg.add(&[FlexType::Undefined]);
        agg.add(&[FlexType::Integer(42)]);
        agg.add(&[FlexType::Integer(99)]);
        assert_eq!(agg.finalize(), FlexType::Integer(42));
    }

    #[test]
    fn test_quantile_median() {
        let mut agg = QuantileAggregator::median();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            agg.add(&[FlexType::Float(v)]);
        }
        match agg.finalize() {
            FlexType::Float(v) => assert!((v - 3.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_quantile_25() {
        let mut agg = QuantileAggregator::new(0.25);
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
            agg.add(&[FlexType::Float(v)]);
        }
        // 25th percentile of [1,2,3,4,5,6,7,8]
        // idx = round((8-1) * 0.25) = round(1.75) = 2 → value 3.0
        match agg.finalize() {
            FlexType::Float(v) => assert!((v - 3.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_aggspec_convenience_constructors() {
        // Just verify they construct without error
        let _ = AggSpec::variance(0, "var");
        let _ = AggSpec::stddev(0, "std");
        let _ = AggSpec::concat(0, "vals");
        let _ = AggSpec::count_distinct(0, "unique_count");
        let _ = AggSpec::quantile(0, 0.5, "median");
        let _ = AggSpec::median(0, "median");
        let _ = AggSpec::select_one(0, "first");
        let _ = AggSpec::frequency_count(0, "freq");
        let _ = AggSpec::zip_list(0, "zipped");
        let _ = AggSpec::vector_sum(0, "vsum");
        let _ = AggSpec::vector_avg(0, "vavg");
    }

    #[test]
    fn test_frequency_count() {
        let mut agg = FrequencyCountAggregator::new();
        agg.add(&[FlexType::String("a".into())]);
        agg.add(&[FlexType::String("b".into())]);
        agg.add(&[FlexType::String("a".into())]);
        agg.add(&[FlexType::String("c".into())]);
        agg.add(&[FlexType::String("a".into())]);
        let result = agg.finalize();
        if let FlexType::Dict(d) = result {
            let a_count = d.iter().find(|(k, _)| k == &FlexType::String("a".into()))
                .map(|(_, v)| v.clone());
            assert_eq!(a_count, Some(FlexType::Integer(3)));
        } else {
            panic!("Expected Dict");
        }
    }

    #[test]
    fn test_zip_list() {
        let mut agg = ZipListAggregator::new();
        agg.add(&[FlexType::Integer(1)]);
        agg.add(&[FlexType::Integer(2)]);
        agg.add(&[FlexType::Integer(3)]);
        let result = agg.finalize();
        if let FlexType::List(l) = result {
            assert_eq!(l.len(), 3);
            assert_eq!(l[0], FlexType::Integer(1));
        } else {
            panic!("Expected List");
        }
    }

    #[test]
    fn test_vector_sum() {
        let mut agg = VectorSumAggregator::new();
        agg.add(&[FlexType::Vector(std::sync::Arc::from(vec![1.0, 2.0, 3.0]))]);
        agg.add(&[FlexType::Vector(std::sync::Arc::from(vec![4.0, 5.0, 6.0]))]);
        let result = agg.finalize();
        if let FlexType::Vector(v) = result {
            assert_eq!(&*v, &[5.0, 7.0, 9.0]);
        } else {
            panic!("Expected Vector");
        }
    }

    #[test]
    fn test_vector_avg() {
        let mut agg = VectorAvgAggregator::new();
        agg.add(&[FlexType::Vector(std::sync::Arc::from(vec![2.0, 4.0]))]);
        agg.add(&[FlexType::Vector(std::sync::Arc::from(vec![4.0, 6.0]))]);
        let result = agg.finalize();
        if let FlexType::Vector(v) = result {
            assert!((v[0] - 3.0).abs() < 1e-10);
            assert!((v[1] - 5.0).abs() < 1e-10);
        } else {
            panic!("Expected Vector");
        }
    }
}
