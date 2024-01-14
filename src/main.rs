use anyhow::Error;
use anyhow::anyhow;
use anyhow::Result;
use opencv::{self as cv, prelude::*};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use opencv::core::Point;
use rand::Rng;

struct PuzzlePiece {
    side1: Vec<Point>,
    side2: Vec<Point>,
    side3: Vec<Point>,
    side4: Vec<Point>,
}

fn main() -> Result<()> {
    my_contour()?;
Ok(())
}

fn my_contour() -> Result<(), anyhow::Error> {
    let file_names = vec![
    // "IMG20240109211005", 
    // "IMG20240113121213", 
    // "IMG20240113121228", 
    // "IMG20240113121241",
    // "IMG20240113121256",
    // "IMG20240113121307",
    // "IMG20240113121330",
    // "IMG20240113121341",
    // "IMG20240113121354",
    // "IMG20240113121410",
    "IMG20240113121426",
    // "IMG20240113121438",
    // "IMG20240113121458",
    // "IMG20240113121510",
    ];

    file_names.into_par_iter().for_each(|file_name| { let _ = process(file_name); });
    // file_names.into_iter().for_each(|file_name| { process(file_name); });

    Ok(())
}

enum Direction {
    UpSide,
    DownSide,
    RightSide,
    LeftSide
}

fn process(file_name: &str) -> Result<(), anyhow::Error> {
    println!("Process: {} ...",file_name);
    let orignial_phase = cv::imgcodecs::imread(format!("./assets/{}.jpg", file_name).as_str(), cv::imgcodecs::IMREAD_COLOR)?;
    let phase = orignial_phase.clone();
    let phase = to_grey(&phase)?;
    let grey_phase = blur(&phase)?;

    let threshold = search_best_threshold(&grey_phase)?;
    let (contours, phase) = sub_process(&grey_phase, threshold)?;
    
    let contours = spli_contour(file_name, &grey_phase, contours)?;
    
    write_contour(format!("{}", file_name).as_str(), &contours)?;
    draw_contour(file_name, &orignial_phase, &contours)?;
    
    Ok(())
}

fn spli_contour(file_name: &str, grey_phase: &Mat, original_contour: cv::types::VectorOfVectorOfPoint) -> Result<cv::types::VectorOfVectorOfPoint, Error> {
    let mut contour_values = cv::types::VectorOfVectorOfPoint::new();

    let sobel_image = sobel(&file_name, &grey_phase, Direction::RightSide)?;
    let single_contours = split_single_contour(&original_contour, &sobel_image, Direction::RightSide)?;
    contour_values.push(single_contours);

    let sobel_image = sobel(&file_name, &grey_phase, Direction::LeftSide)?;
    let single_contours = split_single_contour(&original_contour, &sobel_image, Direction::LeftSide)?;
    contour_values.push(single_contours);
    
    let sobel_image = sobel(&file_name, &grey_phase, Direction::DownSide)?;
    let single_contours = split_single_contour(&original_contour, &sobel_image, Direction::DownSide)?;
    contour_values.push(single_contours);
    
    let sobel_image = sobel(&file_name, &grey_phase, Direction::UpSide)?;
    let single_contours = split_single_contour(&original_contour, &sobel_image, Direction::UpSide)?;
    contour_values.push(single_contours);
    
    Ok(contour_values)
}

fn sobel(file_name: &str, phase: &Mat, direction: Direction) -> Result<Mat, Error> {
    let mut new_phase = cv::core::Mat::default();

    let ddepth = 2; // output image depth, see [filter_depths] “combinations”; in the case of 8-bit input images it will result in truncated derivatives
    let mut dx = 0; // order of the derivative x.
    let mut dy = 0; // order of the derivative y.
    let ksize = 3; // size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    let mut scale = 20.0; // optional scale factor for the computed derivative values; by default, no scaling is applied (see get_deriv_kernels for details).
    let mut delta = -500.0; // optional delta value that is added to the results prior to storing them in dst.
    let border_type = cv::core::BORDER_DEFAULT; // pixel extrapolation method, see #BorderTypes. [BORDER_WRAP] is not supported.
    let mut dir_str = String::new();

    match direction {
        Direction::UpSide => {
            scale = -20.0;
            dy = 1;
            dir_str = "UpSide".to_string();
        },
        Direction::DownSide => {
            scale = 20.0;
            dy = 1;
            dir_str = "DownSide".to_string();
        },
        Direction::RightSide => {
            scale = 20.0;
            dx = 1;
            dir_str = "RightSide".to_string();
        },
        Direction::LeftSide => {
            scale = -20.0;
            dx = 1;
            dir_str = "LeftSide".to_string();
        },

    }

    match cv::imgproc::sobel(&phase,
        &mut new_phase,
        ddepth,
        dx,
        dy,
        ksize,
        scale,
        delta,
        border_type) {
            Ok(image) => {},
            Err(err) => println!("Sobel error {}", err)
        }
    
    cv::imgcodecs::imwrite(format!("{}_sobel_{}.jpg", file_name, dir_str).as_str(), &new_phase, &cv::core::Vector::default())?;
    Ok(new_phase)
}

fn search_best_threshold(grey_phase: &Mat) -> Result<i32, Error> {
    let mut threshold = 0;
    let mut min_len = usize::MAX;
    for threshold_value in 0..255 {
        let (contours_cv, phase) = sub_process(grey_phase, threshold_value)?;

        let mut len = 0;
        for first in &contours_cv {
            len = first.len();
            break;
        }

        if  len > 1000 && len < min_len {
            min_len = len;
            threshold = threshold_value;
        }

    }
    Ok(threshold)
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

fn sub_process(grey_phase: &Mat, threshold_value: i32) -> Result<(cv::types::VectorOfVectorOfPoint, Mat), Error> {
    let phase = threshold(grey_phase, threshold_value)?;
    let phase = bitwise(&phase)?;
    let phase = morph(&phase)?;
    let contour_values = contour(&phase)?;
    Ok((contour_values, phase))
}

/* struct Side {
    x: i32,
    y: i32,
    present: bool,
}

impl Side {
    fn new(x: i32, y: i32, present: bool) -> Self {
        Self { x: x, y: y, present: present}
    }
} */

fn split_single_contour(contours: &cv::types::VectorOfVectorOfPoint, phase: &Mat, direction: Direction) -> std::io::Result<cv::types::VectorOfPoint> {
    // let mut index = 0;
    // let mut side = Vec::<Side>::new();
    let mut vector = cv::types::VectorOfPoint::new();


    let mut cx = 0.0;
    let mut cy = 0.0;

    for first in contours {
        match cv::imgproc::moments_def(&first) {
            Ok(moment) =>         {
                // println!("moment: {:?}", moment);
                // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                cx = moment.m10/moment.m00;
                cy = moment.m01/moment.m00;
            },
            Err(err) => println!("Error: {:?}", err),
        }

        for point in first.iter() {
            match direction {
                Direction::DownSide => {
                    if point.y > cy as i32{
                        vector.push(cv::core::Point::new(point.x, point.y));
                    }
                },
                Direction::UpSide => {
                    if point.y < cy as i32{
                        vector.push(cv::core::Point::new(point.x, point.y));
                    }
                },
                Direction::RightSide => {
                    if point.x > cx as i32{
                        vector.push(cv::core::Point::new(point.x, point.y));
                    }
                },
                Direction::LeftSide => {
                    if point.x < cx as i32 {
                        vector.push(cv::core::Point::new(point.x, point.y));
                    }
                },
            }
            // vector.push(cv::core::Point::new(point.x, point.y));
            // side.push(Side::new(point.x, point.y, true));
        }


        /* for point in first.iter() {
            match phase.at_2d::<u16>(point.y,point.x) {
                Ok(n) => {
                    if *n > 0 {
                        match direction {
                            Direction::DownSide => {
                                if point.y > cy as i32{
                                    vector.push(cv::core::Point::new(point.x, point.y));
                                }
                            },
                            Direction::UpSide => {
                                if point.y < cy as i32{
                                    vector.push(cv::core::Point::new(point.x, point.y));
                                }
                            },
                            Direction::RightSide => {
                                if point.x > cx as i32{
                                    vector.push(cv::core::Point::new(point.x, point.y));
                                }
                            },
                            Direction::LeftSide => {
                                if point.x < cx as i32 {
                                    vector.push(cv::core::Point::new(point.x, point.y));
                                }
                            },
                        }
                        // vector.push(cv::core::Point::new(point.x, point.y));
                        // side.push(Side::new(point.x, point.y, true));
                    }
                    else {
                        // side.push(Side::new(point.x, point.y, false));
                    }
                }
                Err(err) => {println!("Error: {}", err);}
            }
        } */
        break;
    }

    /* let mut output = String::new();

    for side_value in side.iter() {
        output += &format!("{},{} -> {}\n", side_value.x, side_value.y, side_value.present);
    }
    output += "\n\n";


    let mut file = File::create(format!("{}_side.txt", direction))?;
    file.write_all(output.as_bytes())?; */

    Ok(vector)
}

fn write_contour(file_name: &str, contours_cv: &cv::types::VectorOfVectorOfPoint) -> std::io::Result<()> {
    let mut output = String::new();
    for first in contours_cv {
        for point in first.iter() {
            output += &format!("{},{}\n", point.x, point.y);
        }
        output += "\n\n";
    }

    let mut file = File::create(format!("{}_contour.txt", file_name))?;
    file.write_all(output.as_bytes())?;

    Ok(())
}

fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::imgproc::cvt_color(&phase, &mut new_phase, cv::imgproc::COLOR_BGR2GRAY, 0)?;

    Ok(new_phase)
}

fn draw_contour(file_name: &str, orignial_phase: &Mat, contours_cv: &cv::types::VectorOfVectorOfPoint) -> Result<(), anyhow::Error> {
    let mut phase = orignial_phase.clone();
    let zero_offset = cv::core::Point::new(0, 0);
    let thickness: i32 = 20;

    let mut rng = rand::thread_rng();

    for index in 0..contours_cv.len() {
        let n1 = rng.gen_range(0.0..255.0);
        let n2 = rng.gen_range(0.0..255.0);
        let n3 = rng.gen_range(0.0..255.0);
        let color = cv::core::Scalar::new(n1, n2, n3, 255.0);
        cv::imgproc::draw_contours(&mut phase,
            contours_cv,
            index as i32,
            color,
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset)?;
        }

    let name = format!("./{}_contours.jpg", file_name);
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    Ok(())
}

fn contour(phase: &Mat) -> Result<cv::types::VectorOfVectorOfPoint, anyhow::Error> {
    let mut contour_values = cv::types::VectorOfVectorOfPoint::new();
    cv::imgproc::find_contours(
        &phase, 
        &mut contour_values, 
        cv::imgproc:: RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        cv::core::Point::new(0, 0),
    )?;

    Ok(contour_values)
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
