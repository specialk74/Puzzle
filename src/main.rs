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
    /* 
    // Read image
    let img = cv::imgcodecs::imread("./assets/IMG20240108210634.jpg", cv::imgcodecs::IMREAD_COLOR)?;
    // Use Orb
    let mut orb = cv::features2d::ORB::create(
        500,
        1.2,
        8,
        31,
        0,
        2,
        cv::features2d::ORB_ScoreType::HARRIS_SCORE,
        31,
        20,
    )?;
    let mut orb_keypoints = cv::core::Vector::default();
    let mut orb_desc = cv::core::Mat::default();
    let mut dst_img = cv::core::Mat::default();
    let mask = cv::core::Mat::default();
    orb.detect_and_compute(&img, &mask, &mut orb_keypoints, &mut orb_desc, false)?;
    cv::features2d::draw_keypoints(
        &img,
        &orb_keypoints,
        &mut dst_img,
        cv::core::VecN([0., 255., 0., 255.]),
        cv::features2d::DrawMatchesFlags::DEFAULT,
    )?;
    dbg!(orb_keypoints, orb_desc);

    cv::imgproc::rectangle(
        &mut dst_img,
        cv::core::Rect::from_points(cv::core::Point::new(0, 0), cv::core::Point::new(50, 50)),
        cv::core::VecN([255., 0., 0., 0.]),
        -1,
        cv::imgproc::LINE_8,
        0,
    )?;
    // Use SIFT
    let mut sift = cv::features2d::SIFT::create(0, 3, 0.04, 10., 1.6, true)?;
    let mut sift_keypoints = cv::core::Vector::default();
    let mut sift_desc = cv::core::Mat::default();
    sift.detect_and_compute(&img, &mask, &mut sift_keypoints, &mut sift_desc, false)?;
    cv::features2d::draw_keypoints(
        &dst_img.clone(),
        &sift_keypoints,
        &mut dst_img,
        cv::core::VecN([0., 0., 255., 255.]),
        cv::features2d::DrawMatchesFlags::DEFAULT,
    )?;

    dbg!(sift_keypoints, sift_desc);
    // Write image using OpenCV
    cv::imgcodecs::imwrite("./tmp.png", &dst_img, &cv::core::Vector::default())?;
    // Convert :: cv::core::Mat -> ndarray::ArrayView3
    let a = dst_img.try_as_array()?;
    // Convert :: ndarray::ArrayView3 -> RgbImage
    // Note, this require copy as RgbImage will own the data
    let test_image = array_to_image(a);
    // Note, the colors will be swapped (BGR <-> RGB)
  	// Will need to swap the channels before
    // converting to RGBImage
    // But since this is only a demo that
    // it indeed works to convert cv::core::Mat -> ndarray::ArrayView3
    // I'll let it be
    test_image.save("out.png")?;

    //fun_name()?;
*/
    my_contour()?;
Ok(())
}

// IMG20240109211005.jpg
// IMG20240113121213.jpg
// IMG20240113121228.jpg
// IMG20240113121241.jpg
// IMG20240113121256.jpg
// IMG20240113121307.jpg
// IMG20240113121330.jpg
// IMG20240113121341.jpg
// IMG20240113121354.jpg
// IMG20240113121410.jpg
// IMG20240113121426.jpg
// IMG20240113121438.jpg
// IMG20240113121458.jpg
// IMG20240113121510.jpg

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
    //let file_names = vec!["IMG20240109211005"];
    file_names.into_par_iter().for_each(|file_name| { process(file_name); });
    /* for file_name in file_names {
        process(file_name)?;
    } */
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
    // let maxresult = 0;

    // let which_index = 0;
    let mut max_len = 0;
    // let mut index = 0;
    for first in contours_cv {
        /*
        if first.len() > max_len {
            max_len = first.len();
            which_index = index;
        } 
        index += 1; */
        max_len = first.len();
        // println!("{} -> {}", file_name, max_len);
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

    /* let mut point = cv::core::Point::default();
    point.x = 10;
    point.y = 200; */
    /* cv::imgproc::put_text_def(&mut phase,
    format!("Index: {} - MaxLen: {}", which_index, max_len).as_str(),
    point,
    0,
    5.0,
    color
    )?; */
    let name = format!("./{}_contours.jpg", file_name);
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;
    //cv::highgui::imshow(&name, &new_phase)?;
    // cv::highgui::create_trackbar( "track1", &name, Some(threshold_value), 255,  cv::highgui::TrackbarCallback::None)?;
    // println!("Threshold: {}", threshold_value);

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
    // let name = format!("./phase{}_morphology_ex.jpg", count);
    // cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;
    Ok(new_phase)
}

fn bitwise(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::core::bitwise_not_def(
        &phase,
        &mut new_phase
    )?;
    // let name = format!("./phase{}_bitwise_not_def.jpg", count);
    // cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;
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
    //let name = format!("./phase{}_threshold.jpg", count);
    // cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;
    Ok(new_phase)
}

fn my_resize(phase: &Mat, name: &String) -> Result<(), anyhow::Error> {
    let mut clone = cv::core::Mat::default();
    cv::imgproc::resize(
        phase, 
        &mut clone, 
        cv::core::Size::new(0, 0), 
        0.05, 
        0.05, 
        cv::imgproc::INTER_LINEAR)?;
    cv::highgui::imshow(&name, &clone)?;
    Ok(())
}
trait AsArray {
    fn try_as_array(&self) -> Result<ArrayView3<u8>>;
}
impl AsArray for cv::core::Mat {
    fn try_as_array(&self) -> Result<ArrayView3<u8>> {
        if !self.is_continuous() {
            return Err(anyhow!("Mat is not continuous"));
        }
        let bytes = self.data_bytes()?;
        let size = self.size()?;
        let a = ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)?;
        Ok(a)
    }
}
// From Stack Overflow: https://stackoverflow.com/questions/56762026/how-to-save-ndarray-in-rust-as-image
fn array_to_image(arr: ArrayView3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());
let (height, width, _) = arr.dim();
    let raw = arr.to_slice().expect("Failed to extract slice from array");
RgbImage::from_raw(width as u32, height as u32, raw.to_vec())
        .expect("container should have the right size for the image dimensions")
}