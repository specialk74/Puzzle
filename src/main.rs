use anyhow::Error;
use anyhow::anyhow;
use anyhow::Result;
use opencv::{self as cv, prelude::*};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::Write;
use opencv::core::Point;
use rand::Rng;
use strum::{EnumIter, IntoEnumIterator};

struct PuzzlePiece {
    file_name: String,
    contours: cv::types::VectorOfVectorOfPoint,

    x_min: Point,
    y_min: Point,
    x_max: Point,
    y_max: Point,

    original: Mat,
    grey: Mat,
    original_contours: cv::types::VectorOfVectorOfPoint,
    corners: cv::types::VectorOfPoint,

    cx: i32,
    cy: i32,

    left_up_corner: Point,
    left_down_corner: Point,
    right_up_corner: Point,
    right_down_corner: Point,
}

impl PuzzlePiece {
    fn new() -> Self {
        Self {
            file_name: "puzzle".to_string(),
            contours: cv::types::VectorOfVectorOfPoint::default(),

            x_min: Point::new(i32::MAX,0),
            y_min: Point::new(0,i32::MAX),
            x_max: Point::new(0,0),
            y_max: Point::new(0,0),

            original: cv::core::Mat::default(),
            grey: cv::core::Mat::default(),
            original_contours: cv::types::VectorOfVectorOfPoint::default(),
            corners: cv::types::VectorOfPoint::default(),

            cx : 0,
            cy: 0,

            left_up_corner: Point::new(i32::MAX,i32::MAX),
            left_down_corner: Point::new(i32::MAX,0),
            right_up_corner: Point::new(0,i32::MAX),
            right_down_corner: Point::new(0,0),
        }
    }
}

fn main() -> Result<()> {
    my_contour()?;
Ok(())
}

fn my_contour() -> Result<(), anyhow::Error> {
    let file_names = vec![
    "IMG20240109211005", 
    "IMG20240113121213", 
    "IMG20240113121228", 
    "IMG20240113121241",
    "IMG20240113121256",
    "IMG20240113121307",
    "IMG20240113121330",
    "IMG20240113121341",
    "IMG20240113121354",
    // "IMG20240113121410",
    "IMG20240113121426",
    "IMG20240113121438",
    "IMG20240113121458",
    "IMG20240113121510",
    ];

    file_names.into_par_iter().for_each(|file_name| { let _ = process(file_name); });
    // file_names.into_iter().for_each(|file_name| { process(file_name); });

    Ok(())
}

#[derive(EnumIter, Debug)]
enum Direction {
    UpSide,
    DownSide,
    RightSide,
    LeftSide
}

fn process(file_name: &str) -> Result<(), anyhow::Error> {
    println!("Process: {} ...",file_name);
    let mut puzzle = PuzzlePiece::new();
    puzzle.file_name = file_name.to_string();
    puzzle.original = cv::imgcodecs::imread(format!("./assets/{}.jpg", puzzle.file_name).as_str(), cv::imgcodecs::IMREAD_COLOR)?;
    
    let phase = to_grey(&puzzle.original)?;
    let grey_phase = blur(&phase)?;

    let threshold = search_best_threshold(&grey_phase)?;
    let (contours, phase) = sub_process(&grey_phase, threshold)?;
    puzzle.original_contours = contours;

    let Ok((cx, cy)) = find_centroid(&puzzle) else { todo!() };
    puzzle.cx = cx;
    puzzle.cy = cy;

    let _ = find_min_max(&mut puzzle);
    
    let phase = fill_poly(&puzzle)?;
    let corners = find_corners(&puzzle, &phase)?;
    set_corners(&mut puzzle, &corners);
    puzzle.corners = corners;

    let _ = draw_simple_contour(&puzzle);


    puzzle.contours = split_contour(&grey_phase, &mut puzzle)?;

    //write_contour(&puzzle)?;
    draw_contour(&puzzle)?;
    
    Ok(())
}

fn set_corners(puzzle: &mut PuzzlePiece, corners: &cv::types::VectorOfPoint) {
    for point in corners.iter() {
        if point.y < puzzle.cy {
            if point.x < puzzle.cx {
                puzzle.left_up_corner.x = point.x;
                puzzle.left_up_corner.y = point.y;
            }
            else {
                puzzle.right_up_corner.x = point.x;
                puzzle.right_up_corner.y = point.y;
            }
        }
        else {
            if point.x < puzzle.cx {
                puzzle.left_down_corner.x = point.x;
                puzzle.left_down_corner.y = point.y;
            }
            else {
                puzzle.right_down_corner.x = point.x;
                puzzle.right_down_corner.y = point.y;
            }
        }
    }
}

fn fill_poly(puzzle: &PuzzlePiece)-> Result<Mat, anyhow::Error> {
    let white = cv::core::Scalar::new(255.0, 255.0, 255.0, 255.0);
    let black = cv::core::Scalar::new(0.0, 0.0, 0.0, 255.0);

    let mut new_phase = cv::core::Mat::new_size_with_default(
        puzzle.original.size()?,
        cv::core::CV_8UC1,
        white, 
    )?;

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &puzzle.original_contours,
        black,
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_poly: {}", err)
    }

    let name = format!("./{}_fill_poly.jpg", puzzle.file_name);
    cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    Ok(new_phase)
}

fn split_contour(grey_phase: &Mat, puzzle: &PuzzlePiece) -> Result<cv::types::VectorOfVectorOfPoint, Error> {
    let mut contour_values = cv::types::VectorOfVectorOfPoint::new();

    for dir in Direction::iter() {
        let single_contours = split_single_contour(&puzzle, dir)?;
        contour_values.push(single_contours);
    }
    
    Ok(contour_values)
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
    let contour_values = find_contour(&phase)?;
    Ok((contour_values, phase))
}

fn find_centroid(puzzle: &PuzzlePiece) -> Result<(i32, i32), Error> {
    let mut cx = 0.0;
    let mut cy = 0.0;

    for first in &puzzle.original_contours {
        match cv::imgproc::moments_def(&first) {
            Ok(moment) =>         {
                // println!("moment: {:?}", moment);
                // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                cx = moment.m10/moment.m00;
                cy = moment.m01/moment.m00;
            },
            Err(err) => println!("Error: {:?}", err),
        }
        break;
    }

    // let mut new_phase = cv::core::Mat::default();
    // new_phase = puzzle.original.clone();
    // let color = cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0);
    // let point = Point::new(cx as i32, cy as i32);

    // match cv::imgproc::circle (
    //     &mut new_phase,
    //     point,
    //     20,
    //     color,
    //     cv::imgproc::FILLED,
    //     cv::imgproc::LINE_8,
    //     0
    // ) {
    //     Ok(_) => {},
    //     Err(err) => { return Err(anyhow!(err));}
    // }

    // let name = format!("./{}_centroid.jpg", puzzle.file_name);
    // cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    Ok((cx as i32, cy as i32))
}

fn get_polygon(puzzle: &PuzzlePiece, delta: i32, direction: Direction) -> cv::types::VectorOfPoint {
    let mut polygon = cv::types::VectorOfPoint::new();
    polygon.push(cv::core::Point::new(puzzle.cx, puzzle.cy));

    match direction {
        Direction::DownSide => {
            polygon.push(puzzle.left_down_corner.clone());
            polygon.push(cv::core::Point::new(puzzle.left_down_corner.x, 
                puzzle.left_down_corner.y + (puzzle.y_max.y - puzzle.left_down_corner.y) + delta));
            polygon.push(cv::core::Point::new(puzzle.right_down_corner.x, 
                puzzle.right_down_corner.y + (puzzle.y_max.y - puzzle.right_down_corner.y) + delta));
            polygon.push(puzzle.right_down_corner.clone());
        },
        Direction::UpSide => {
            polygon.push(puzzle.left_up_corner.clone());
            polygon.push(cv::core::Point::new(puzzle.left_up_corner.x, 
                puzzle.left_up_corner.y - (puzzle.left_up_corner.y - puzzle.y_min.y) -delta));
            polygon.push(cv::core::Point::new(puzzle.right_up_corner.x, 
                puzzle.right_up_corner.y - (puzzle.right_up_corner.y - puzzle.y_min.y) -delta));
            polygon.push(puzzle.right_up_corner.clone());
        },
        Direction::RightSide => {
            polygon.push(puzzle.right_up_corner.clone());
            polygon.push(cv::core::Point::new(puzzle.right_up_corner.x 
                + (puzzle.x_max.x - puzzle.right_up_corner.x) + delta, puzzle.right_up_corner.y));
            polygon.push(cv::core::Point::new(puzzle.right_down_corner.x 
                + (puzzle.x_max.x - puzzle.right_down_corner.x) + delta, puzzle.right_down_corner.y));
            polygon.push(puzzle.right_down_corner.clone());
        },
        Direction::LeftSide => {
            polygon.push(puzzle.left_up_corner.clone());
            polygon.push(cv::core::Point::new(puzzle.left_up_corner.x
                - (puzzle.left_up_corner.x - puzzle.x_min.x) - delta, puzzle.left_up_corner.y));
            polygon.push(cv::core::Point::new(puzzle.left_down_corner.x
                - (puzzle.left_down_corner.x - puzzle.x_min.x) - delta, puzzle.left_down_corner.y));
            polygon.push(puzzle.left_down_corner.clone());
        },
    }

    polygon
}

fn split_single_contour(puzzle: &PuzzlePiece, direction: Direction) -> std::io::Result<cv::types::VectorOfPoint> {
    let mut vector = cv::types::VectorOfPoint::new();

    for first in &puzzle.original_contours {
        
        let mut polygon = get_polygon(puzzle, 100, direction);

        for point in first.iter() {
            match cv::imgproc::point_polygon_test(
                &polygon,
                cv::core::Point2f::new(point.x as f32, point.y as f32),
                true){
                    Ok(val) => {
                        if val >= 0.0 {
                            //println!("{:?} -> x: {}, y: {} -> val: {}", direction, point.x, point.y, val);
                            vector.push(cv::core::Point::new(point.x, point.y));  
                        }
                    },
                    Err(err) => {println!("Error on split_single_contour: {}", err);}
                }
        }

        break;
    }

    Ok(vector)
}

fn find_corners(puzzle: &PuzzlePiece, phase: &Mat)-> Result<cv::types::VectorOfPoint, anyhow::Error> {

    let mut corners = cv::types::VectorOfPoint::new();
    let max_corners = 4;
    let quality_level = 0.1;
    let min_distance = 1300.0;
    let mask = &puzzle.original_contours;
    let mut block_size: i32 = 100;
    let use_harris_detector: bool = false;
    let k: f64 = 0.1;

    loop {
        match cv::imgproc::good_features_to_track(
            &phase,
            &mut corners,
            max_corners,
            quality_level,
            min_distance,
            &cv::core::no_array(),
            block_size,
            use_harris_detector,
            k
        ){
            Ok(_) => {},
            Err(err) => println!("Error on find_corners: {}", err)
        }

        if corners.len() == 4 {
            // let mut new_phase = cv::core::Mat::default();
            // new_phase = puzzle.original.clone();
            // let color = cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0);

            // for point in corners.iter() {
            //     cv::imgproc::circle (
            //         &mut new_phase,
            //         point,
            //         20,
            //         color,
            //         cv::imgproc::FILLED,
            //         cv::imgproc::LINE_8,
            //         0
            //     )?;
            // }

            // let name = format!("./{}_find_corners.jpg", puzzle.file_name);
            // cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;
            break;
        }
        block_size += 1;

        //cv::highgui::wait_key(300);
        //println!("block_size: {}", block_size);

        if block_size > 150 {
            println!("Exit from find_corners for {}.jpg", puzzle.file_name);
            break;
        }
    }

    Ok(corners)
}

fn write_contour(puzzle: &PuzzlePiece) -> std::io::Result<()> {

    let mut output = String::new();
    for first in &puzzle.contours {
        for point in first.iter() {
            output += &format!("{},{}\n", point.x, point.y);
        }
        output += "\n\n";
    }

    let mut file = File::create(format!("{}_contour.txt", puzzle.file_name))?;
    file.write_all(output.as_bytes())?;

    Ok(())
}

fn find_min_max(puzzle: &mut PuzzlePiece) -> std::io::Result<()> {
    for first in &puzzle.original_contours {
        for point in first.iter() {
            if point.x < puzzle.x_min.x {
                puzzle.x_min.x = point.x;
                puzzle.x_min.y = point.y;
            }

            if point.x > puzzle.x_max.x {
                puzzle.x_max.x = point.x;
                puzzle.x_max.y = point.y;
            }

            if point.y < puzzle.y_min.y {
                puzzle.y_min.x = point.x;
                puzzle.y_min.y = point.y;
            }

            if point.y > puzzle.y_max.y {
                puzzle.y_max.x = point.x;
                puzzle.y_max.y = point.y;
            }
        }
    }

    Ok(())
}

fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    match cv::imgproc::cvt_color(&phase, 
        &mut new_phase, 
        cv::imgproc::COLOR_BGR2GRAY, 
        0){
            Ok(_) => {},
            Err(err) => {
                println!("To Grey Error: {}", err);
                return Err(anyhow!(err));
            }
        }

    Ok(new_phase)
}

fn draw_contour(puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {

    let mut phase = puzzle.original.clone();
    let zero_offset = cv::core::Point::new(0, 0);
    let thickness: i32 = 20;

    let mut rng = rand::thread_rng();

    for index in 0..puzzle.contours.len() {
        let n1 = rng.gen_range(0.0..255.0);
        let n2 = rng.gen_range(0.0..255.0);
        let n3 = rng.gen_range(0.0..255.0);
        let color = cv::core::Scalar::new(n1, n2, n3, 255.0);
        match cv::imgproc::draw_contours(&mut phase,
            &puzzle.contours,
            index as i32,
            color,
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset){
                Ok(_) => {},
                Err(err) => {
                    println!("Error on draw_contours - index {}: {}", index, err);
                    return Err(anyhow!(err));
                }
            }
    }

    for dir in Direction::iter() {
        let n1 = rng.gen_range(0.0..255.0);
        let n2 = rng.gen_range(0.0..255.0);
        let n3 = rng.gen_range(0.0..255.0);
        let color = cv::core::Scalar::new(n1, n2, n3, 255.0);

        let polygon = get_polygon(puzzle, 100, dir);
        cv::imgproc::polylines(
            &mut phase,
            &polygon,
            true,
            cv::core::Scalar::new(255.0, 0.0, 0.0, 255.0),
            10,
            cv::imgproc::LINE_8,
            0
        )?;
    }

    let name = format!("./{}_contours.jpg", puzzle.file_name);
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    Ok(())
}

fn draw_simple_contour(puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {

    let mut phase = puzzle.original.clone();
    let zero_offset = cv::core::Point::new(0, 0);
    let thickness: i32 = 20;

    let mut rng = rand::thread_rng();

    for index in 0..puzzle.original_contours.len() {
        let n1 = rng.gen_range(0.0..255.0);
        let n2 = rng.gen_range(0.0..255.0);
        let n3 = rng.gen_range(0.0..255.0);
        let color = cv::core::Scalar::new(n1, n2, n3, 255.0);
        match cv::imgproc::draw_contours(&mut phase,
            &puzzle.original_contours,
            index as i32,
            color,
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset){
                Ok(_) => {},
                Err(err) => {
                    println!("Error on draw_contours - index {}: {}", index, err);
                    return Err(anyhow!(err));
                }
            }
            break;
    }

    let color = cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0);
    for point in puzzle.corners.iter() {
        cv::imgproc::circle (
            &mut phase,
            point,
            20,
            color,
            cv::imgproc::FILLED,
            cv::imgproc::LINE_8,
            0
        )?;
    }

    let point = Point::new(puzzle.cx, puzzle.cy);
    cv::imgproc::circle (
        &mut phase,
        point,
        20,
        color,
        cv::imgproc::FILLED,
        cv::imgproc::LINE_8,
        0
    )?;
    
    let name = format!("./{}_simple_contours.jpg", puzzle.file_name);
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    Ok(())
}

fn find_contour(phase: &Mat) -> Result<cv::types::VectorOfVectorOfPoint, anyhow::Error> {
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
