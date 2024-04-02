// Implementation of Pixelated Image Abstraction (PIA)
// Link: https://pixl.cs.princeton.edu/pubs/Gerstner_2012_PIA/index.php

use image::{DynamicImage, Rgba, GenericImage};
use nalgebra::iter;

use super::super::Pixelizer;
use palette::{rgb::Rgb, FromColor, IntoColor, Lab, Srgb};
use core::num;
use std::{collections::HashSet, sync::mpsc::Sender};

pub struct PIAPixelizer{
    temperature_init: Option<f32>,
    temperature_final: f32, // default should be 1.0
    m : f32, // coefficient for the cost function, eq (1) of the paper,
    alpha: f32, // efault should be 0.7
    epsilon_palette: f32, // default is 1
    epsilon_cluster: f32, // default is 0.25
    post_process_saturation: f32, // default is 1.1 in the paper
}

impl PIAPixelizer {
    // m is set to 45 in the paper
    // post_process_saturation is set to 1.1 in the paper
    pub fn new(temperature_init: Option<f32>, temperature_final: f32, m: f32, alpha: f32, epsilon_palette: f32, epsilon_cluster: f32, post_process_saturation: f32) -> Self {
        Self {
            temperature_init,
            temperature_final,
            m,
            alpha,
            epsilon_palette,
            epsilon_cluster,
            post_process_saturation
        }
    }
}

#[derive(Debug)]
struct SuperPixel{
    position: (f32, f32),
    palette_color: Lab,
    pixels: HashSet<(u32, u32)>,
    original_position: (f32, f32),
    p_s: f32,
    p_c: Vec<f32>,
    sp_color: Lab
}

fn lab_distance(lab1: &Lab, lab2: &Lab) -> f32{
    let l_diff = lab1.l - lab2.l;
    let a_diff = lab1.a - lab2.a;
    let b_diff = lab1.b - lab2.b;

    (l_diff * l_diff + a_diff * a_diff + b_diff * b_diff).sqrt()
}

impl SuperPixel{
    fn new(position: (f32, f32), palette_color: Lab, num_sp: usize) -> Self{
        let pixels = HashSet::new();
        let original_position = position;
        let p_s = 1.0 / num_sp as f32;
        let sp_color = Lab::default();

        let p_c = vec![0.5; 2];

        Self{
            position,
            palette_color,
            pixels,
            original_position,
            p_s,
            p_c,
            sp_color
        }
    }

    // eq (1) of the paper
    fn cost(&self, x: u32, y: u32, img_lab : Vec<Vec<Lab>>, m: f32, nb_pixels_in : u32, nb_pixels_out : u32) -> f32{
        let lab_color = img_lab[y as usize][x as usize];
        let d_lab = lab_distance(&self.palette_color, &lab_color);
        let d_spatial = ((self.position.0 - x as f32).powi(2) + (self.position.1 - y as f32).powi(2)).sqrt();

        d_lab + m * ((nb_pixels_out as f32) / (nb_pixels_in as f32)).sqrt() * d_spatial
    }

    fn add_pixel(&mut self, x: u32, y: u32){
        self.pixels.insert((x, y));
    }

    fn clear_pixels(&mut self){
        self.pixels.clear();
    }

    fn update_pos(&mut self){
        assert!(self.pixels.len() > 0);
        self.position = self.pixels.iter().fold((0.0, 0.0), |acc, (x, y)| {
            (acc.0 + *x as f32, acc.1 + *y as f32)
        });
        self.position = (self.position.0 / self.pixels.len() as f32, self.position.1 / self.pixels.len() as f32);
    }

    fn update_sp_color(&mut self, img: &Vec<Vec<Lab>>){
        assert!(self.pixels.len() > 0);
        self.sp_color = self.pixels.iter().fold(Lab::default(), |acc, (x, y)| {
            acc+img[*y as usize][*x as usize]
        }) / self.pixels.len() as f32;
    }

    fn normalize_probs(&mut self){
        let sum = self.p_c.iter().sum::<f32>();
        self.p_c = self.p_c.iter().map(|x| x / sum).collect();
    }

    fn update_palette_color(&mut self, colors: &Vec<Lab>){
        let highest_prob = self.p_c.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        self.palette_color = colors[highest_prob];
    }
}


#[derive(Debug)]
struct Palette{
    colors: Vec<Lab>,
    probabilities: Vec<f32> // invariants: the sum of the probabilities is 1
}

impl Palette{
    fn new(init_color : Lab) -> Self{
        let mut colors = vec![init_color; 2];
        colors[1] = Lab::from_components((colors[1].l + 1.0, colors[1].a+1.0, colors[1].b+1.0));
        let probabilities = vec![0.5; 2];
        Self{
            colors,
            probabilities
        }
    }

    fn conditional_prob(&self, color: Lab, temperature: f32) -> Vec<f32>{
        let mut probs = Vec::with_capacity(self.colors.len());
        for k in 0..self.colors.len(){
            let d_lab = lab_distance(&self.colors[k], &color);
            let prob = self.probabilities[k]*(-d_lab/temperature).exp();
            probs.push(prob);
        }
        probs
    }
}

struct PIAGlobal{
    superpixels: Vec<Vec<SuperPixel>>,
    clusters: Vec<(usize, usize)>,
    palette: Palette,
}

impl PIAGlobal{
    fn new(w: u32, h: u32, lab_img: &Vec<Vec<Lab>>) -> Self{
        let num_sp = (w * h) as usize;
        let mut superpixels = Vec::with_capacity(num_sp);

        let width_orig = lab_img[0].len();
        let height_orig = lab_img.len();

        let width_ratio = width_orig as f32 / (w+1) as f32;
        let height_ratio = height_orig as f32 / (h+1) as f32;

        // get average color
        let sum_color = lab_img.iter().map(|row| {
            row.iter().fold((0.0, 0.0, 0.0), |acc, color| {
                (acc.0 + color.l, acc.1 + color.a, acc.2 + color.b)
            })
            }).fold((0.0, 0.0, 0.0), |acc, color| {
                (acc.0 + color.0, acc.1 + color.1, acc.2 + color.2)
            });
        let total_pixels = (width_orig * height_orig) as f32;
        let avg_lab_color = Lab::new(sum_color.0 / total_pixels, sum_color.1 / total_pixels, sum_color.2 / total_pixels);

        for y in 1..(h+1){
            let mut row_superpixels = Vec::with_capacity(w as usize);
            for x in 1..(w+1){
                let position = (x as f32 * width_ratio, y as f32 * height_ratio);
                row_superpixels.push(SuperPixel::new(position, avg_lab_color, num_sp));
            }
            superpixels.push(row_superpixels);
        }
        let clusters = vec![(0, 1); 1];
        let palette = Palette::new(avg_lab_color);
        Self{
            superpixels,
            clusters,
            palette
        }
    }

    fn refine_superpixels(&mut self, img: &Vec<Vec<Lab>>, m: f32){
        for superpixel in self.superpixels.iter_mut().flatten(){
            superpixel.clear_pixels();
        }

        // Find best superpixel for each pixel
        for y in 0..img.len(){
            for x in 0..img[0].len(){
                let mut min_cost = std::f32::MAX;
                let mut min_sp_index = (0, 0);
                for (y_sp, superpixel_row) in self.superpixels.iter().enumerate(){
                    for (x_sp, superpixel) in superpixel_row.iter().enumerate(){
                        let cost = superpixel.cost(x as u32, y as u32, img.to_vec(), m, 1, 1);
                        if cost < min_cost{
                            min_cost = cost;
                            min_sp_index = (x_sp, y_sp);
                        }
                    }
                }
                self.superpixels[min_sp_index.1][min_sp_index.0].add_pixel(x as u32, y as u32);
            }
        }

        // Refine colors and positions for superpixels
        for superpixel in self.superpixels.iter_mut().flatten(){
            superpixel.update_pos();
            superpixel.update_sp_color(img);
        }

        // TODO: Laplacian smoothing

        // TODO: Bilateral filtering
    }

    fn associate(&mut self, temperature: f32){
        self.superpixels.iter_mut().flatten().for_each(|superpixel| {
            let probs = self.palette.conditional_prob(superpixel.sp_color, temperature);
            superpixel.p_c = probs;
            superpixel.normalize_probs();
            superpixel.update_palette_color(&self.palette.colors);
        });

        self.palette.probabilities = vec![0.0; self.palette.probabilities.len()];
        for superpixel in self.superpixels.iter().flatten(){
            for (k, prob) in superpixel.p_c.iter().enumerate(){
                self.palette.probabilities[k] += prob * superpixel.p_s;
            }
        }

        println!("{:?}", self.palette.probabilities);
        println!("{}", self.palette.probabilities.iter().sum::<f32>());
    }

    fn refine_palette(&mut self, epsilon_palette: &f32) -> bool{
        let mut total_change: f32 = 0.0;
        let mut updated_colors = Vec::with_capacity(self.palette.colors.len());
        for (k, color) in self.palette.colors.iter_mut().enumerate(){
            let palette_probability = self.palette.probabilities[k];
            let acc = self.superpixels.iter().flatten().fold((0., 0., 0.), |acc, superpixel| {
                let prob = superpixel.p_c[k] * superpixel.p_s / palette_probability;
                (acc.0 + superpixel.sp_color.l * prob, acc.1 + superpixel.sp_color.a * prob, acc.2 + superpixel.sp_color.b * prob)
            });
            let new_color = Lab::new(acc.0, acc.1, acc.2);
            let change = lab_distance(&new_color, color);
            updated_colors.push(new_color);
            total_change += change;
        }

        self.palette.colors = updated_colors;

        return total_change < *epsilon_palette; //TODO: check whether I should not divide by the number of colors? 
    }

    fn expand_palette(&mut self,num_colors: &usize, epsilon_cluster: &f32){
        let cluster_size = self.clusters.len();
        if cluster_size < *num_colors{
            for k in 0..cluster_size{
                if (self.clusters.len() < *num_colors){
                    let (i, j) = self.clusters[k];
                    let color_i = self.palette.colors[i];
                    let color_j = self.palette.colors[j];
                    let prob_i = self.palette.probabilities[i]/2.0;
                    let prob_j = self.palette.probabilities[j]/2.0;
                    if (lab_distance(&color_i, &color_j) > *epsilon_cluster){
                        self.palette.colors.push(color_i);
                        self.palette.colors.push(color_j);
                        self.palette.probabilities.push(prob_i);
                        self.palette.probabilities.push(prob_j);
                        self.palette.probabilities[i] = prob_i;
                        self.palette.probabilities[j] = prob_j;
                        // let new_color = Lab::new((color_i.l + color_j.l) / 2.0, (color_i.a + color_j.a) / 2.0, (color_i.b + color_j.b) / 2.0);
                        // self.palette.colors.push(new_color);
                        self.clusters.push((i, self.palette.colors.len() - 2));
                        self.clusters[k] = (j, self.palette.colors.len() - 1);
                    }
                }
            }
        }
        let cluster_size = self.clusters.len();
        if cluster_size >= *num_colors{
            let mut new_palette_colors = Vec::with_capacity(*num_colors);
            let mut new_palette_probs = Vec::with_capacity(*num_colors);
            for k in 0..cluster_size{
                let (i, j) = self.clusters[k];
                let color_i = self.palette.colors[i];
                let color_j = self.palette.colors[j];
                let prob_i = self.palette.probabilities[i];
                let prob_j = self.palette.probabilities[j];
                let new_color = Lab::new((color_i.l + color_j.l) / 2.0, (color_i.a + color_j.a) / 2.0, (color_i.b + color_j.b) / 2.0);
                new_palette_colors.push(new_color);
                new_palette_probs.push(prob_i + prob_j);
            }
            self.palette.colors = new_palette_colors;
            self.palette.probabilities = new_palette_probs;
        } else{
            for k in 0..cluster_size{
                //TODO
                // self.palette.colors[self.clusters[k].1] += perturb;
                let current_color = self.palette.colors[self.clusters[k].1];
                let perturbed_color = Lab::from_components((current_color.l + 1.0, current_color.a+1.0, current_color.b+1.0));
                self.palette.colors[self.clusters[k].1] = perturbed_color;
            }
        }
    }

    fn get_img(&self, saturation: &f32) -> DynamicImage{
        let width = self.superpixels[0].len();
        let height = self.superpixels.len();
        let mut img = DynamicImage::new_rgb8(width as u32, height as u32);
        for (y, row) in self.superpixels.iter().enumerate(){
            for (x, superpixel) in row.iter().enumerate(){
                let (l, mut a, mut b) = superpixel.palette_color.into_components();
                a *= *saturation;
                b *= *saturation;
                let color = Lab::new(l, a, b);
                let srgb_color: Srgb<u8> = Srgb::from_color(color).into_format();
                let rgba_color = Rgba([srgb_color.red, srgb_color.green, srgb_color.blue, 255]);
                img.put_pixel(x as u32, y as u32, rgba_color);
            }
        }
        img
    }

    fn get_current_nb_colors(&self) -> usize{
        self.clusters.len()
    }
}


impl Pixelizer for PIAPixelizer{
    fn pixelize(&self, img: &DynamicImage, width : &u32, height: &u32,  num_colors : &usize) -> DynamicImage{
        let rgb_image = img.to_rgb8();
        let width_orig = rgb_image.width();
        let height_orig = rgb_image.height();
        let lab_vec = rgb_image.enumerate_pixels().map(|(x, y, pixel)| {
            let rgb_color = Srgb::new(pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
                .into_linear();        
            let lab_color: Lab = rgb_color.into_color();
    
            (x, y, lab_color)
        }).fold(vec![vec![Lab::default(); width_orig as usize]; height_orig as usize], | mut acc, (x, y, pixel)| {
            acc[y as usize][x as usize] = pixel;
            return acc;
        });

        let mut global = PIAGlobal::new(*width, *height, &lab_vec);

        let mut temperature = match self.temperature_init {
            Some(t) => t,
            None => {
                35.0 //TODO: PCA here
            }
        };

        let mut iter = 0;
        while temperature > self.temperature_final{
            
            println!("Iteration: {}", iter);
            iter += 1;
            // refine superpixels
            global.refine_superpixels(&lab_vec, self.m);
            // associate
            global.associate(temperature);
            // refine colors in the palette
            let to_expand = global.refine_palette(&self.epsilon_palette);

            // if palette converged, reduce temperature and expand palette if necessary
            if to_expand{
                println!("Expanding palette");
                temperature *= self.alpha;
                if global.get_current_nb_colors() < *num_colors{
                    global.expand_palette(num_colors, &self.epsilon_cluster);
                }
            }
            else {
                println!("Not expanding palette");
            }
        }
        // post process
        let img_out = global.get_img(&self.post_process_saturation);

        // return image
        img_out
    }
}


#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageError};

    use crate::pixelizer::pia::PIAPixelizer;
    use super::Pixelizer;

    #[test]
    fn test_size() {
        let img = image::open("examples/images/ferris_3d_pixelized.png").unwrap();
        let pixelizer = PIAPixelizer{
            temperature_init: Some(35.0),
            temperature_final: 1.0,
            m: 45.0,
            alpha: 0.7,
            epsilon_palette: 1.0,
            epsilon_cluster: 0.25,
            post_process_saturation: 1.1
        };
        let width = 32;
        let height = 32;
        let num_colors = 6;
        let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

        let path_out = "examples/images/ferris_3d_PIA.png";
        pixelized.save(path_out).unwrap();

        let size_pixelized = (pixelized.width(), pixelized.height());
        assert_eq!(size_pixelized, (width, height));
    }

    fn equal_dynamic_image(img1: &DynamicImage, img2: &DynamicImage) -> Result<bool, ImageError>{
        if img1.width() != img2.width() || img1.height() != img2.height(){
            return Ok(false);
        }
        let img_buf_1 = img1.to_rgb8();
        let img_buf_2 = img2.to_rgb8();
        Ok(img_buf_1.as_raw() == img_buf_2.as_raw())
    }

    // #[test]
    // fn test_uniform(){
    //     // Uniform image should be pixelized exactly the same, if given the same width and height
    //     let img = image::open("examples/images/uniform.png").unwrap();
    //     let pixelizer = super::KmeansPixelizer{
    //         num_runs: 8,
    //         max_iter: 20,
    //         color_type: ColorType::Rgb
    //     };
    //     let width = img.width();
    //     let height = img.height();
    //     let num_colors = 8;
    //     let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

    //     let path_out = "examples/images/uniform_pixelized.png";
    //     pixelized.save(path_out).unwrap();

    //     let size_pixelized = (pixelized.width(), pixelized.height());
    //     assert_eq!(size_pixelized, (width, height));
    //     if let Ok(b) = equal_dynamic_image(&img, &pixelized){
    //         assert!(b, "the uniform image is not pixelized to itself");
    //     }
    // }
}