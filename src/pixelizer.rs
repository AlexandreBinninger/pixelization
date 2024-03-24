extern crate image;
use image::{DynamicImage};
use rand::Rng;

pub enum CropMethod{
    NoCrop,
    CropEqual,
    CropRandom
}

pub fn scale_to_size<'a>(img: DynamicImage, scale: &u32, crop: CropMethod) -> (DynamicImage, (u32, u32)){
    let (w, h) = (img.width()/scale, img.height()/scale);
    match crop {
        CropMethod::NoCrop => (img, (w, h)),
        CropMethod::CropEqual => {
            let (new_w, new_h) = (w*scale, h*scale);
            let left = (img.width() - new_w) / 2;
            let top = (img.height() - new_h) / 2;
            (img.crop_imm(left, top, new_w, new_h), (w, h))
        },
        CropMethod::CropRandom => {
            let mut rng = rand::thread_rng();
            let (new_w, new_h) = (w*scale, h*scale);
            let left = rng.gen_range(0..(img.width() - new_w));
            let top = rng.gen_range(0..(img.height() - new_h));
            (img.crop_imm(left, top, new_w, new_h), (w, h))
        }
    }
}

pub trait Pixelizer{
    fn pixelize(&self, img: &DynamicImage, width : &u32, height: &u32,  num_colors : &usize) -> DynamicImage;
    // todo: pixelize, with the "block size" --> computing the width and height, maybe cropping, then call the above pixelize fn
}

pub mod kmeans_pixelizer;