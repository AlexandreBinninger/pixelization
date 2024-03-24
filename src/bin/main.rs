use std::error::Error;

use clap::{Command, Arg, ArgGroup, ArgAction, ValueEnum, value_parser};
use image::imageops::FilterType;
use pixelization::{ColorType, KmeansPixelizer, Pixelizer, scale_to_size, CropMethod};

use minifb::{Key, Window, WindowOptions};

#[derive(ValueEnum, Clone, Debug)]
enum PixelizationMethod{
    Kmeans,
    PIA
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
            .help("Tell which pixelization method to use. Valid are \"kmeans\"")
            .value_parser(value_parser!(PixelizationMethod))
            .default_value("kmeans")
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
            .value_name("UINT")
    )
        .group(ArgGroup::new("dimension")
            .args(&["scale", "size"])
            .required(true))
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap(); // Safe to use unwrap() because it's required
    let output = matches.get_one::<String>("output");
    let n_colors = *matches.get_one::<usize>("n_colors").unwrap();

    println!("Using input file: {}", input_path);

    let img_src = image::open(input_path)?;


    let (img, (w, h)) = 
    if matches.contains_id("scale"){
        println!("Using scale");
        let scale = matches.get_one::<u32>("scale").unwrap();
        scale_to_size(img_src, scale, CropMethod::NoCrop) //TODO: different crop method
    }
    else{ // if matches.contains_id("size"){
        println!("Using size");
        (img_src, parse_size(matches.get_one::<String>("size").unwrap())?)
    };

    let img_pixelized = match matches.get_one::<PixelizationMethod>("method") {
        Some(PixelizationMethod::Kmeans) => {
            let kmeans_pixelizer = KmeansPixelizer::new(5, 20, ColorType::Lab); //TODO: make it depending on arguments
            kmeans_pixelizer.pixelize(&img, &w, &h, &n_colors)
        },
        Some(PixelizationMethod::PIA) => return Err("PIA method is not implemented".into()),
        None => return Err("Method is not implemented".into())
    };


    // Show
    if *matches.get_one::<bool>("show").unwrap_or(&false){
        println!("Press ESC to quit.");
        let img_pixelized = img_pixelized.resize(w*4, h*4, FilterType::Nearest);
        let img_rgb = img_pixelized.to_rgb8();
        let width = img_rgb.width();
        let height = img_rgb.height();
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
    
        // Display loop
        while window.is_open() && !window.is_key_down(Key::Escape) {
            window
                .update_with_buffer(&buffer, width as usize, height as usize)
                .expect("Failed to update window");
        }
    }

    // Save
    if let Some(output) = output {
        println!("Output will be saved to: {}", output);
        img_pixelized.save(output)?;
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