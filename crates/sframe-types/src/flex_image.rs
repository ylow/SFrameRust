//! Image type for ML workflows.
//!
//! Provides a basic image representation compatible with the C++ SFrame
//! `flex_image` type. Images are stored as raw pixel data with metadata
//! about dimensions, channels, and format.

use std::sync::Arc;

/// Pixel format for image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// Raw pixels, one byte per channel.
    Raw,
    /// JPEG-compressed data.
    Jpeg,
    /// PNG-compressed data.
    Png,
}

/// A flexible image type for ML workflows.
///
/// Stores image pixel data along with dimension and format metadata.
/// This type is used for image columns in SFrame datasets,
/// supporting common ML data pipeline operations.
///
/// # Example
/// ```
/// use sframe_types::flex_image::{FlexImage, ImageFormat};
///
/// // Create a 2x2 RGB image
/// let pixels = vec![
///     255, 0, 0,    // red pixel
///     0, 255, 0,    // green pixel
///     0, 0, 255,    // blue pixel
///     255, 255, 0,  // yellow pixel
/// ];
/// let img = FlexImage::new(pixels, 2, 2, 3, ImageFormat::Raw);
/// assert_eq!(img.width(), 2);
/// assert_eq!(img.height(), 2);
/// assert_eq!(img.channels(), 3);
/// assert_eq!(img.num_pixels(), 4);
/// ```
#[derive(Clone)]
pub struct FlexImage {
    /// Raw pixel data (or compressed data for JPEG/PNG).
    data: Arc<[u8]>,
    /// Image width in pixels.
    width: usize,
    /// Image height in pixels.
    height: usize,
    /// Number of channels (1=grayscale, 3=RGB, 4=RGBA).
    channels: usize,
    /// Pixel/compression format.
    format: ImageFormat,
    /// Image version (for format evolution).
    version: u8,
}

impl FlexImage {
    /// Create a new image from raw pixel data.
    pub fn new(
        data: Vec<u8>,
        width: usize,
        height: usize,
        channels: usize,
        format: ImageFormat,
    ) -> Self {
        FlexImage {
            data: Arc::from(data),
            width,
            height,
            channels,
            format,
            version: 0,
        }
    }

    /// Create an empty (zero-sized) image.
    pub fn empty() -> Self {
        FlexImage {
            data: Arc::from(Vec::new()),
            width: 0,
            height: 0,
            channels: 0,
            format: ImageFormat::Raw,
            version: 0,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of channels (1=grayscale, 3=RGB, 4=RGBA).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Pixel/compression format.
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    /// Total number of pixels.
    pub fn num_pixels(&self) -> usize {
        self.width * self.height
    }

    /// Raw data bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Data length in bytes.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Whether the image is empty (zero dimensions).
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Get pixel value at (x, y) for a given channel.
    /// Returns None if out of bounds or if format is compressed.
    pub fn pixel(&self, x: usize, y: usize, channel: usize) -> Option<u8> {
        if self.format != ImageFormat::Raw {
            return None;
        }
        if x >= self.width || y >= self.height || channel >= self.channels {
            return None;
        }
        let idx = (y * self.width + x) * self.channels + channel;
        self.data.get(idx).copied()
    }

    /// Resize the image metadata (does not resample pixels).
    /// Use this when decoding compressed formats where the actual
    /// pixel dimensions differ from stored metadata.
    pub fn with_dimensions(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }
}

impl std::fmt::Debug for FlexImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FlexImage({}x{} {:?} {}ch {}B)",
            self.width,
            self.height,
            self.format,
            self.channels,
            self.data.len()
        )
    }
}

impl std::fmt::Display for FlexImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Image({}x{} {}ch)",
            self.width, self.height, self.channels
        )
    }
}

impl PartialEq for FlexImage {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.channels == other.channels
            && self.format == other.format
            && self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_image() {
        let pixels = vec![255, 0, 0, 0, 255, 0];
        let img = FlexImage::new(pixels, 2, 1, 3, ImageFormat::Raw);
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.num_pixels(), 2);
        assert_eq!(img.data_len(), 6);
        assert!(!img.is_empty());
    }

    #[test]
    fn test_pixel_access() {
        let pixels = vec![10, 20, 30, 40, 50, 60];
        let img = FlexImage::new(pixels, 2, 1, 3, ImageFormat::Raw);
        assert_eq!(img.pixel(0, 0, 0), Some(10));
        assert_eq!(img.pixel(0, 0, 2), Some(30));
        assert_eq!(img.pixel(1, 0, 0), Some(40));
        assert_eq!(img.pixel(2, 0, 0), None); // out of bounds
    }

    #[test]
    fn test_empty_image() {
        let img = FlexImage::empty();
        assert!(img.is_empty());
        assert_eq!(img.num_pixels(), 0);
    }

    #[test]
    fn test_image_display() {
        let img = FlexImage::new(vec![0; 12], 2, 2, 3, ImageFormat::Raw);
        assert_eq!(format!("{}", img), "Image(2x2 3ch)");
    }
}
