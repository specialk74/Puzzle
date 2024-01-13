use anyhow::Error;
use anyhow::anyhow;
use anyhow::Result;
use image::RgbImage;
use ndarray::ArrayView3;
use opencv::{self as cv, prelude::*};
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::i32::MAX;
use std::io::Write;
use rand::Rng;

fn main() -> Result<()> {
    my_contour()?;
Ok(())
}

fn my_contour() -> Result<(), anyhow::Error> {
    let file_names = vec!["IMG20240109211005", 
    "IMG20240113121213", 
    "IMG20240113121228", 
    "IMG20240113121241",
    "IMG20240113121256",
    "IMG20240113121307",
    "IMG20240113121330",
    "IMG20240113121341",
    "IMG20240113121354",
    "IMG20240113121410",
    "IMG20240113121426",
    "IMG20240113121438",
    "IMG20240113121458",
    "IMG20240113121510"];

    file_names.into_par_iter().for_each(|file_name| { process(file_name); });

    Ok(())
}

fn process(file_name: &str) -> Result<(), anyhow::Error> {
    println!("Process: {} ...",file_name);
    let orignial_phase = cv::imgcodecs::imread(format!("./assets/{}.jpg", file_name).as_str(), cv::imgcodecs::IMREAD_COLOR)?;
    let phase = orignial_phase.clone();
    let phase = to_grey(&phase)?;
    let grey_phase = blur(&phase)?;

    let mut threshold = 0;
    let mut min_len = usize::MAX;
    for threshold_value in 0..255 {
        let contours_cv = sub_process(&grey_phase, threshold_value)?;

        let mut len = 0;
        for first in &contours_cv {
            len = first.len();
            break;
        }
    
        if  len > 2000 && len < min_len {
            min_len = len;
            threshold = threshold_value;
        }

    }
    let contours_cv = sub_process(&grey_phase, threshold)?;
    match draw_contour(file_name, &orignial_phase, &contours_cv) {
        Ok(()) => write_contour(file_name, &contours_cv)?,
        _ => {}
    }
    
    Ok(())
}

fn blur(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let ksize = cv::core::Size::new(10,10);
    let mut new_phase = cv::core::Mat::default();

    cv::imgproc::blur_def (
        &phase,
        &mut new_phase,
        ksize
    )?;
    Ok(new_phase)
}

fn sub_process(grey_phase: &Mat, threshold_value: i32) -> Result<cv::core::Vector<cv::core::Vector<cv::core::Point_<i32>>>, Error> {
    let phase = threshold(grey_phase, threshold_value)?;
    let phase = bitwise(&phase)?;
    let phase = morph(&phase)?;
    let contours_cv = contour(&phase)?;
    Ok(contours_cv)
}

fn write_contour(file_name: &str, contours_cv: &cv::core::Vector<cv::core::Vector<cv::core::Point_<i32>>>) -> std::io::Result<()> {
    let mut output = String::new();

    for first in contours_cv {
        for point in first.iter() {
            output += &format!("{},{}\n", point.x, point.y);
        }
        output += &format!("\n\n\n\n");
    }

    let mut file = File::create(format!("{}_contour.txt", file_name))?;
    file.write_all(output.as_bytes())
}

fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::imgproc::cvt_color(&phase, &mut new_phase, cv::imgproc::COLOR_BGR2GRAY, 0)?;

    Ok(new_phase)
}

fn draw_contour(file_name: &str, orignial_phase: &Mat, contours_cv: &cv::core::Vector<cv::core::Vector<cv::core::Point_<i32>>>) -> Result<(), anyhow::Error> {
    let mut phase = orignial_phase.clone();
    let zero_offset = cv::core::Point::new(0, 0);
    let thickness: i32 = 5;

    let mut max_len = 0;
    for first in contours_cv {
        max_len = first.len();
        break;
    }

    if  max_len < 1000 {
        return Err(anyhow!("..."));
    }

    let color = cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0);
    cv::imgproc::draw_contours(&mut phase,
        contours_cv,
        -1,
        color,
        thickness,
        cv::imgproc::LINE_8,
        &cv::core::no_array(),
        0,
        zero_offset)?;

    let name = format!("./{}_contours.jpg", file_name);
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    Ok(())
}

fn contour(phase: &Mat) -> Result<cv::core::Vector<cv::core::Vector<cv::core::Point_<i32>>>, anyhow::Error> {
    let mut contours_cv = cv::types::VectorOfVectorOfPoint::new();
    cv::imgproc::find_contours(
        &phase, 
        &mut contours_cv, 
        cv::imgproc:: RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        cv::core::Point::new(0, 0),
    )?;

    Ok(contours_cv)
}

fn morph(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    let anchor = cv::core::Point::new(-1, -1);
    let ksize = cv::core::Size::new(5,5);
    let kernel = cv::imgproc::get_structuring_element(
        0,
        ksize,
        anchor
    )?;
    cv::imgproc::morphology_ex(
        &phase,
        &mut new_phase,
        cv::imgproc::MORPH_OPEN,
        &kernel,
        anchor,
        1,
        cv::core::BORDER_CONSTANT,
        cv::imgproc::morphology_default_border_value()?)?;

    Ok(new_phase)
}

fn bitwise(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::core::bitwise_not_def(
        &phase,
        &mut new_phase
    )?;

    Ok(new_phase)
}

fn threshold(phase: &Mat, threshold_value: i32) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::imgproc::threshold(
        &phase,
        &mut new_phase,
        threshold_value as f64,
        255.0,
        cv::imgproc::THRESH_BINARY
    )?;

    Ok(new_phase)
}
