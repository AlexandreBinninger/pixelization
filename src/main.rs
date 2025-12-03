use std::error::Error;
use std::path::{Path, PathBuf};

use clap::{Command, Arg, ArgGroup, ArgAction, ValueEnum, value_parser};
use image::{DynamicImage, imageops::FilterType};
use pixelization::{ColorType, KmeansPixelizer, PIAPixelizer, Pixelizer, scale_to_size, CropMethod};

use minifb::{Key, KeyRepeat, Window, WindowOptions};
// use rayon::iter::IntoParallelIterator;

#[derive(ValueEnum, Clone, Debug)]
enum PixelizationMethod{
    Kmeans,
    PIA
}

#[derive(ValueEnum, Clone, Debug)]
enum ColorSpace{
    Lab,
    Rgb
}

#[derive(ValueEnum, Clone, Debug)]
enum CropMethodArg{
    NoCrop,
    CropEqual,
    CropRandom
}

fn main() -> Result<(), Box<dyn Error>>{
    let matches = Command::new("pixelize")
        .version("0.1")
        .author("Alexandre Binninger")
        .about("Pixelize the input image.")
        .arg(
            Arg::new("input")
                .help("Sets the input file to use")
                .required(true)
                .value_name("FILE")
                .index(1),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .value_parser(value_parser!(PathBuf))
                .help("Sets an optional output file")
        )
        .arg(
            Arg::new("show")
            .help("Display the result in another window.")
            .long("show")
            .action(ArgAction::SetTrue)
        )
        .arg(Arg::new("method")
            .action(ArgAction::Set)
            .long("method")
            .value_name("PixelizationMethod")
            .help("Tell which pixelization method to use. Valid are \"kmeans\" and \"pia\".")
            .value_parser(value_parser!(PixelizationMethod))
            .default_value("pia")
    )
        .arg(Arg::new("scale")
            .short('s')
            .long("scale")
            .help("The downscale factor.")
            .value_name("UINT")
            .value_parser(value_parser!(u32))
            .conflicts_with("size"),
        )
        .arg(Arg::new("size")
            .long("size")
            .help("The size of the output.")
            .value_name("W,H")
            .conflicts_with("scale"),
        )
        .arg(Arg::new("n_colors")
            .long("n_colors")
            .short('k')
            .help("The number of colors used for the output.")
            .default_value("8")
            .value_parser(value_parser!(usize))
            .value_name("UINT"))
        .arg(Arg::new("num_runs")
            .long("num_runs")
            .short('r')
            .help("The number of runs for the kmeans algorithm.")
            .default_value("3")
            .value_parser(value_parser!(u32))
            .value_name("UINT"))
        .arg(Arg::new("max_iter")
            .long("max_iter")
            .short('i')
            .help("The maximum number of iterations for the kmeans algorithm.")
            .default_value("20")
            .value_parser(value_parser!(usize))
            .value_name("UINT"))
        .arg(Arg::new("color_space")
            .long("color_space")
            .short('c')
            .help("The color space to use.")
            .action(ArgAction::Set)
            .value_name("ColorSpace")
            .value_parser(value_parser!(ColorSpace))
            .default_value("lab"))
        .arg(Arg::new("crop_method")
            .long("crop_method")
            .help("The crop method to use when scaling. Possible values are no-crop, crop-equal and crop-random.")
            .action(ArgAction::Set)
            .value_name("CropMethod")
            .value_parser(value_parser!(CropMethodArg))
            .default_value("no-crop"))
        .arg(Arg::new("verbose")
            .short('v')
            .long("verbose")
            .help("Prints debug information verbosely.")
            .action(ArgAction::SetTrue))
        .group(ArgGroup::new("dimension")
            .args(&["scale", "size"])
            .required(true))
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap(); // Safe to use unwrap() because it's required
    let input_path = Path::new(input_path);
    let output = matches.get_one::<PathBuf>("output");
    let n_colors = *matches.get_one::<usize>("n_colors").unwrap();

    println!("Using input file: {}", input_path.display());

    let img_src = image::open(input_path)?;

    let (img, (w, h)) = 
    if matches.contains_id("scale"){
        println!("Using scale argument.");
        let scale = matches.get_one::<u32>("scale").unwrap();
        let crop_method = match matches.get_one::<CropMethodArg>("crop_method").unwrap() {
            CropMethodArg::NoCrop => CropMethod::NoCrop,
            CropMethodArg::CropEqual => CropMethod::CropEqual,
            CropMethodArg::CropRandom => CropMethod::CropRandom
        };
        scale_to_size(img_src, scale, crop_method)
    }
    else{ // if matches.contains_id("size"){
        println!("Using size argument.");
        (img_src, parse_size(matches.get_one::<String>("size").unwrap())?)
    };
    println!("Target size: {}x{}", w, h);

    let img_pixelized = match matches.get_one::<PixelizationMethod>("method") {
        Some(PixelizationMethod::Kmeans) => {
            println!("Using Kmeans pixelization.");
            let num_runs = *matches.get_one::<u32>("num_runs").unwrap();
            let max_iter = *matches.get_one::<usize>("max_iter").unwrap();
            let color_space = matches.get_one::<ColorSpace>("color_space").unwrap();
            let color_type = match color_space {
                ColorSpace::Lab => ColorType::Lab,
                ColorSpace::Rgb => ColorType::Rgb
            };
            let kmeans_pixelizer = KmeansPixelizer::new(num_runs, max_iter, color_type);
            kmeans_pixelizer.pixelize(&img, w, h, n_colors)
        },
        Some(PixelizationMethod::PIA) => {
            let mut pia_pixelizer = PIAPixelizer::default();
            let verbose = *matches.get_one::<bool>("verbose").unwrap_or(&false);
            pia_pixelizer.set_verbose(verbose);
            pia_pixelizer.pixelize(&img, w, h, n_colors)
        },
        None => return Err("Method is not implemented".into())
    };

    let img_pixelized = match img_pixelized {
        Ok(img) => img,
        Err(e) => return Err(format!("Pixelization error: {:?}", e).into()),
    };

    let output_path = if let Some(path) = matches.get_one::<PathBuf>("output"){
        img_pixelized.save(path)?;
        path.clone()
    } else{
        let file_stem = input_path.file_stem().unwrap_or_default();
        let mut new_name = file_stem.to_os_string();
        new_name.push("_pixelized.png");
        input_path.with_file_name(new_name)
    };

    // Save
    if let Some(output) = output {
        // Show
        if *matches.get_one::<bool>("show").unwrap_or(&false){
            show(&img, &img_pixelized, output.to_str().unwrap());
        }
    } else{
        show(&img, &img_pixelized, output_path.to_str().unwrap());
    }

    Ok(())
}



fn parse_size(size_str : &str) -> Result<(u32, u32), &'static str>{
    let parts: Vec<&str> = size_str.split(',').collect();
    if parts.len() != 2{
        return Err("Provide size in the format w,h.");
    }

    let w = parts[0].parse::<u32>().map_err(|_| "Invalid width.")?;
    let h = parts[1].parse::<u32>().map_err(|_| "Invalid height.")?;

    match w {
        0 => Err("Width is zero."),
        _ => match h {
            0 => Err("Height is zero."),
            _ => Ok((w, h))
        }
    }
}

fn show(img: &DynamicImage, img_pixelized: &DynamicImage, output_name : &str){
    println!("Press ESC to quit.");
    println!("Press ENTER to alternate between input and pixelization.");
    println!("Press S to save the pixelized image to {}", output_name);
    let width = img.width();
    let height = img.height();
    let img_pixelized_resized = img_pixelized.resize_exact(width, height, FilterType::Nearest);
    let img_pixelized_rgb = img_pixelized_resized.to_rgb8();
    let buffer_pixelized: Vec<u32> = img_pixelized_rgb.pixels().map(|p| {
        let [r, g, b] = p.0;
        // Assuming the alpha channel is fully opaque (0xFF)
        (r as u32) << 16 | (g as u32) << 8 | (b as u32) | 0xFF000000
    }).collect();

    let img_rgb = img.to_rgb8();
    let buffer: Vec<u32> = img_rgb.pixels().map(|p| {
        let [r, g, b] = p.0;
        // Assuming the alpha channel is fully opaque (0xFF)
        (r as u32) << 16 | (g as u32) << 8 | (b as u32) | 0xFF000000
    }).collect();

    let mut window = Window::new(
        "Pixelization",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )
    .expect("Failed to create window");
    let mut current_buffer = &buffer_pixelized;
    // Display loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
        .update_with_buffer(current_buffer, width as usize, height as usize)
        .expect("Failed to update window");
        if window.is_key_pressed(Key::Enter, KeyRepeat::No){
            current_buffer = if current_buffer == &buffer { &buffer_pixelized} else { &buffer};
        }
        if window.is_key_pressed(Key::S, KeyRepeat::No){
            img_pixelized.save(output_name).expect("Failed to save image.");
            println!("Image saved to {}", output_name);
        }
    }
}