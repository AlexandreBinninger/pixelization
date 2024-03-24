extern crate image;
use image::{DynamicImage};

pub trait Pixelizer{
    fn pixelize(&self, img: &DynamicImage, width : &u32, height: &u32,  num_colors : &usize) -> DynamicImage;
    // todo: pixelize, with the "block size" --> computing the width and height, maybe cropping, then call the above pixelize fn
}

pub mod kmeans_pixelizer;