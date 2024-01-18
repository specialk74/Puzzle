use anyhow::Error;
use anyhow::anyhow;
use anyhow::Result;
use ndarray::Dimension;
use opencv::{self as cv, prelude::*};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use opencv::core::Point;
use rand::Rng;

struct PuzzlePiece {
    side1: cv::types::VectorOfPoint,
    side2: cv::types::VectorOfPoint,
    side3: cv::types::VectorOfPoint,
    side4: cv::types::VectorOfPoint,

    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,

    original: Mat,
    grey: Mat,
    contours: cv::types::VectorOfVectorOfPoint,
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
            side1: cv::types::VectorOfPoint::default(),
            side2: cv::types::VectorOfPoint::default(),
            side3: cv::types::VectorOfPoint::default(),
            side4: cv::types::VectorOfPoint::default(),

            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,

            original: cv::core::Mat::default(),
            grey: cv::core::Mat::default(),
            contours: cv::types::VectorOfVectorOfPoint::default(),
            corners: cv::types::VectorOfPoint::default(),

            cx : 0,
            cy: 0,

            left_up_corner: Point::new(0,0),
            left_down_corner: Point::new(0,0),
            right_up_corner: Point::new(0,0),
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
    // "IMG20240109211005.jpg", 
    // "IMG20240113121213.jpg", 
    // "IMG20240113121228.jpg", 
    // "IMG20240113121241.jpg",
    // "IMG20240113121256.jpg",
    // "IMG20240113121307.jpg",
    // "IMG20240113121330.jpg",
    // "IMG20240113121341.jpg",
    // "IMG20240113121354.jpg",
    // "IMG20240113121410.jpg",
    //"IMG20240117211141.dng",
    "IMG20240113121426.jpg",
    // "IMG20240113121438.jpg",
    // "IMG20240113121458.jpg",
    // "IMG20240113121510.jpg",
    ];

    file_names.into_par_iter().for_each(|file_name| { let _ = process(file_name); });
    // file_names.into_iter().for_each(|file_name| { process(file_name); });

    Ok(())
}

#[derive(Debug)]
enum Direction {
    UpSide,
    DownSide,
    RightSide,
    LeftSide
}

fn process(file_name: &str) -> Result<(), anyhow::Error> {
    println!("Process: {} ...",file_name);
    let mut puzzle_piece = PuzzlePiece::new();
    puzzle_piece.original = cv::imgcodecs::imread(format!("./assets/{}", file_name).as_str(), cv::imgcodecs::IMREAD_COLOR)?;
    
    let phase = puzzle_piece.original.clone();
    let phase = to_grey(&phase)?;
    let grey_phase = blur(&phase)?;

    let threshold = search_best_threshold(&grey_phase)?;
    let (contours, phase) = sub_process(&grey_phase, threshold)?;
    puzzle_piece.contours = contours;
    
    let phase = fill_poly(&grey_phase, &puzzle_piece.contours)?;
    let corners = good_features_to_track(&puzzle_piece, &phase)?;
    puzzle_piece.corners = corners;

    let contours = spli_contour(file_name, &grey_phase, &puzzle_piece.contours)?;

    write_contour(format!("{}", file_name).as_str(), &puzzle_piece.contours)?;
    draw_contour(file_name, &puzzle_piece)?;
    
    Ok(())
}

fn fill_poly(phase: &Mat, contour: &cv::types::VectorOfVectorOfPoint)-> Result<Mat, anyhow::Error> {
    let white = cv::core::Scalar::new(255.0, 255.0, 255.0, 255.0);
    let black = cv::core::Scalar::new(0.0, 0.0, 0.0, 255.0);
    //let line_type: i32 = 2;
    //let shift: i32 = 0;

    let mut new_phase = cv::core::Mat::new_size_with_default(
        phase.size()?,
        cv::core::CV_8UC1,
        white, 
    )?;

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &contour,
        black,
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_poly: {}", err)
    }

    let name = format!("./fill_poly.jpg");
    cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    println!("Fill_poly ends");

    Ok(new_phase)
}

fn fill_convex_poly(phase: &Mat, contour: &cv::types::VectorOfVectorOfPoint)-> Result<Mat, anyhow::Error> {
    let color = cv::core::Scalar::new(0.0, 0.0, 0.0, 255.0);
    //let line_type: i32 = 2;
    //let shift: i32 = 0;

    let mut new_phase = cv::core::Mat::default();
    new_phase = phase.clone();

    match cv::imgproc::fill_convex_poly_def(
        &mut new_phase,
        &contour,
        color,
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_convex_poly: {}", err)
    }

    let name = format!("./fill_convex_poly.jpg");
    cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    println!("Fill_convex_poly ends");

    Ok(new_phase)
}

fn dilate(phase: &Mat) -> Result<Mat, anyhow::Error> {
    
    /* let shape = 3;
    let ksize = cv::core::Size::new(3,3);
    let anchor = cv::core::Point::new(1, 1);
println!("dilate1");
    if let Ok(mat) = cv::imgproc::get_structuring_element(
        shape,
        ksize,
        anchor
    ) {
        println!("dilate2");
        let mut new_phase = cv::core::Mat::default();
        cv::imgproc::dilate_def (
            &phase,
            &mut new_phase,
            &mat     
        )?;
        println!("dilate3");
    
        let name = format!("./dilate.jpg");
        cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;
        println!("dilate4");
    
        return Ok(new_phase);
    }

    println!("dilate error");
    Err(anyhow!("dilate")) */

    let mut new_phase = cv::core::Mat::default();
    let mut kernel = cv::core::Mat::default();
    cv::imgproc::dilate_def (
        &phase,
        &mut new_phase,
        &kernel
    )?;

    let name = format!("./dilate.jpg");
    cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    Ok(new_phase)
}


fn corner(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let block_size = 30;
    let ksize = 5;
    let k = 10.04;
    let mut new_phase = cv::core::Mat::default();

    match cv::imgproc::corner_harris_def (
        &phase,
        &mut new_phase,
        block_size,
        ksize,
        k
    ){
        Ok(new_phase) => {},
        Err(err) => println!("Erro on corner: {}", err)
    }

    let name = format!("./corner.jpg");
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    println!("Corner end");
    Ok(new_phase)
}

fn spli_contour(file_name: &str, grey_phase: &Mat, original_contour: &cv::types::VectorOfVectorOfPoint) -> Result<cv::types::VectorOfVectorOfPoint, Error> {
    let mut contour_values = cv::types::VectorOfVectorOfPoint::new();

    // let sobel_image = sobel(&file_name, &grey_phase, Direction::RightSide)?;
    let single_contours = split_single_contour(original_contour, Direction::RightSide)?;
    contour_values.push(single_contours);

    // let sobel_image = sobel(&file_name, &grey_phase, Direction::LeftSide)?;
    let single_contours = split_single_contour(original_contour, Direction::LeftSide)?;
    contour_values.push(single_contours);
    
    // let sobel_image = sobel(&file_name, &grey_phase, Direction::DownSide)?;
    let single_contours = split_single_contour(original_contour, Direction::DownSide)?;
    contour_values.push(single_contours);
    
    // let sobel_image = sobel(&file_name, &grey_phase, Direction::UpSide)?;
    let single_contours = split_single_contour(original_contour, Direction::UpSide)?;
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

fn find_centroid(puzzle: &PuzzlePiece) -> (i32, i32) {
    let mut cx = 0.0;
    let mut cy = 0.0;

    for first in &puzzle.contours {
        match cv::imgproc::moments_def(&first) {
            Ok(moment) =>         {
                // println!("moment: {:?}", moment);
                // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                cx = moment.m10/moment.m00;
                cy = moment.m01/moment.m00;
            },
            Err(err) => println!("Error: {:?}", err),
        }
    }

    (cx as i32, cy as i32)
}

fn split_single_contour(contours: &cv::types::VectorOfVectorOfPoint, direction: Direction) -> std::io::Result<cv::types::VectorOfPoint> {
    // let mut index = 0;
    // let mut side = Vec::<Side>::new();

    //println!("split_single_contour: {:?} start...", direction);
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

        dbg!(cx);
        dbg!(cy);
        let mut polygon = cv::types::VectorOfPoint::new();
        polygon.push(cv::core::Point::new(cx as i32, cy as i32));

        // cv::core::Point::new(1062,1034)
        // cv::core::Point::new(2370,3149)
        // cv::core::Point::new(729,2968)
        // cv::core::Point::new(2542,1232)

        let delta = 1000;

        match direction {
            Direction::DownSide => {
                polygon.push(cv::core::Point::new(729, 2968));
                polygon.push(cv::core::Point::new(729, 2968+delta));
                polygon.push(cv::core::Point::new(2370, 3149+delta));
                polygon.push(cv::core::Point::new(2370, 3149));
            },
            Direction::UpSide => {
                polygon.push(cv::core::Point::new(1062, 1034));
                polygon.push(cv::core::Point::new(1062, 1034-delta));
                polygon.push(cv::core::Point::new(2542, 1232-delta));
                polygon.push(cv::core::Point::new(2542, 1232));
            },
            Direction::RightSide => {
                polygon.push(cv::core::Point::new(2370, 3149));
                polygon.push(cv::core::Point::new(2370 + delta, 3149));
                polygon.push(cv::core::Point::new(2542 +delta, 1232));
                polygon.push(cv::core::Point::new(2542, 1232));
            },
            Direction::LeftSide => {
                polygon.push(cv::core::Point::new(1062, 1034));
                polygon.push(cv::core::Point::new(1062-delta, 1034));
                polygon.push(cv::core::Point::new(729-delta, 2968));
                polygon.push(cv::core::Point::new(729, 2968));
            },
        }

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

    /* let mut output = String::new();

    for side_value in side.iter() {
        output += &format!("{},{} -> {}\n", side_value.x, side_value.y, side_value.present);
    }
    output += "\n\n";


    let mut file = File::create(format!("{}_side.txt", direction))?;
    file.write_all(output.as_bytes())?; */
    //println!("split_single_contour: {:?} ends", direction);

    Ok(vector)
}

fn good_features_to_track(puzzle: &PuzzlePiece, phase: &Mat)-> Result<cv::types::VectorOfPoint, anyhow::Error> {

    let mut corners = cv::types::VectorOfPoint::new();
    let max_corners = 4;
    let quality_level = 0.1;
    let min_distance = 1300.0;
    let mask = &puzzle.contours;
    let mut block_size: i32 = 52;
    let use_harris_detector: bool = false;
    let k: f64 = 0.1;

        println!("good_features_to_track - block_size: {}", block_size);
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
            Err(err) => println!("Error on good_features_to_track: {}", err)
        }

        if corners.len() == 4 {
            //dbg!(&corners);
            let mut new_phase = cv::core::Mat::default();
            new_phase = puzzle.original.clone();
            let color = cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0);

            for point in corners.iter() {
                cv::imgproc::circle (
                    &mut new_phase,
                    point,
                    20,
                    color,
                    cv::imgproc::FILLED,
                    cv::imgproc::LINE_8,
                    0
                )?;
            }

            let name = format!("./good_features_to_track.jpg");
            cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;
        }

    dbg!(&corners);
    println!("good_features_to_track ends");

    Ok(corners)
}


fn keypoints(phase: &Mat)-> Result<cv::types::VectorOfKeyPoint, anyhow::Error> {

    let mut corners = cv::types::VectorOfKeyPoint::new();

    match cv::features2d::fast(
        &phase,
        &mut corners,
        0,
        false
    ){
        Ok(_) => {},
        Err(err) => println!("Error on keypoints: {}", err)
    }

    dbg!(&corners);

    let mut new_phase = cv::core::Mat::default();
    new_phase = phase.clone();
    let color = cv::core::Scalar::new(255.0, 255.0, 255.0, 255.0);

    for point in corners.iter() {
        let x = point.pt().x as i32;
        let y = point.pt().y as i32;

        cv::imgproc::circle (
            &mut new_phase,
            cv::core::Point::new(x, y),
            50,
            color,
            cv::imgproc::FILLED,
            cv::imgproc::LINE_8,
            0
        )?;
    }

    let name = format!("./keypoints.jpg");
    cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;
    

    println!("keypoints ends");

    Ok(corners)
}

fn write_contour(file_name: &str, contours_cv: &cv::types::VectorOfVectorOfPoint) -> std::io::Result<()> {

    let mut up_right = cv::core::Point::new(0,i32::MAX);
    let mut down_right = cv::core::Point::new(0,0);

    let mut up_left = cv::core::Point::new(i32::MAX,i32::MAX);
    let mut down_left = cv::core::Point::new(i32::MAX,0);

    let mut x_min = cv::core::Point::new(i32::MAX,0);
    let mut y_min = cv::core::Point::new(0, i32::MAX);
    let mut x_max = cv::core::Point::new(0,0);
    let mut y_max = cv::core::Point::new(0,0);

    let mut output = String::new();
    for first in contours_cv {
        for point in first.iter() {
            output += &format!("{},{}\n", point.x, point.y);

            if point.x < x_min.x {
                x_min.x = point.x;
                x_min.y = point.y;
            }

            if point.x > x_max.x {
                x_max.x = point.x;
                x_max.y = point.y;
            }

            if point.y < y_min.y {
                y_min.x = point.x;
                y_min.y = point.y;
            }

            if point.y > y_max.y {
                y_max.x = point.x;
                y_max.y = point.y;
            }



            if point.x > up_right.x && point.y < up_right.y {
                up_right.x = point.x;
                up_right.y = point.y;
            }
            if point.x > down_right.x && point.y > down_right.y {
                down_right.x = point.x;
                down_right.y = point.y;
            }

            if point.x < up_left.x && point.y < up_left.y {
                up_left.x = point.x;
                up_left.y = point.y;
            }
            if point.x < down_left.x && point.y > down_left.y {
                down_left.x = point.x;
                down_left.y = point.y;
            }
        }

        output += "\n\n";
    }

    // println!("up_right:{:?} - up_left:{:?} - down_right:{:?} - down_left:{:?}", up_right, up_left, down_right, down_left);
    // println!("x_min:{:?} - x_max:{:?} - y_min:{:?} - y_max:{:?}", x_min, x_max, y_min, y_max);

    let mut file = File::create(format!("{}_contour.txt", file_name))?;
    file.write_all(output.as_bytes())?;

    Ok(())
}

fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    println!("To Grey phase...");
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

fn draw_contour(file_name: &str, puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {

    let mut phase = puzzle.original.clone();
    let zero_offset = cv::core::Point::new(0, 0);
    let thickness: i32 = 20;

    println!("draw_contour");

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

    /* 
    let delta = 0;
    cv::imgproc::line(
        &mut phase,
        cv::core::Point::new(1062 - delta,1034 - delta), // up_right
        cv::core::Point::new(2370+delta,3149+delta), // down_left
        cv::core::Scalar::new(255.0, 0.0, 0.0, 255.0),
        5,
        cv::imgproc::LINE_8,
        0
    )?;

    cv::imgproc::line(
        &mut phase,
        cv::core::Point::new(729-delta,2968+delta), // up_left
        cv::core::Point::new(2542+delta,1232-delta), // down_right
        cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0),
        5,
        cv::imgproc::LINE_8,
        0
    )?;
    */

    let mut polygon = cv::types::VectorOfPoint::new();
    let cx = 1660;
    let cy = 2152;

        // cv::core::Point::new(1062,1034)
        // cv::core::Point::new(2370,3149)
        // cv::core::Point::new(729,2968)
        // cv::core::Point::new(2542,1232)

        let delta = 500;
        polygon.clear();
        polygon.push(cv::core::Point::new(cx as i32, cy as i32));
        polygon.push(cv::core::Point::new(729, 2968));
        polygon.push(cv::core::Point::new(729, 2968+delta));
        polygon.push(cv::core::Point::new(2370, 3149+delta));
        polygon.push(cv::core::Point::new(2370, 3149));

        cv::imgproc::polylines(
            &mut phase,
            &polygon,
            true,
            cv::core::Scalar::new(255.0, 255.0, 0.0, 255.0),
            10,
            cv::imgproc::LINE_8,
            0
        )?;


        polygon.clear();
        polygon.push(cv::core::Point::new(cx as i32, cy as i32));
        polygon.push(cv::core::Point::new(1062, 1034));
        polygon.push(cv::core::Point::new(1062, 1034-delta));
        polygon.push(cv::core::Point::new(2542, 1232-delta));
        polygon.push(cv::core::Point::new(2542, 1232));
        cv::imgproc::polylines(
            &mut phase,
            &polygon,
            true,
            cv::core::Scalar::new(0.0, 0.0, 0.0, 255.0),
            10,
            cv::imgproc::LINE_8,
            0
        )?;

        polygon.clear();
        polygon.push(cv::core::Point::new(cx as i32, cy as i32));
        polygon.push(cv::core::Point::new(2370, 3149));
        polygon.push(cv::core::Point::new(2370 + delta, 3149));
        polygon.push(cv::core::Point::new(2542 +delta, 1232));
        polygon.push(cv::core::Point::new(2542, 1232));
        cv::imgproc::polylines(
            &mut phase,
            &polygon,
            true,
            cv::core::Scalar::new(255.0, 0.0, 255.0, 255.0),
            10,
            cv::imgproc::LINE_8,
            0
        )?;

        polygon.clear();
        polygon.push(cv::core::Point::new(cx as i32, cy as i32));
        polygon.push(cv::core::Point::new(1062, 1034));
        polygon.push(cv::core::Point::new(1062-delta, 1034));
        polygon.push(cv::core::Point::new(729-delta, 2968));
        polygon.push(cv::core::Point::new(729, 2968));
        cv::imgproc::polylines(
            &mut phase,
            &polygon,
            true,
            cv::core::Scalar::new(0.0, 255.0, 255.0, 255.0),
            10,
            cv::imgproc::LINE_8,
            0
        )?;
    



    println!("draw_contour -> create file");
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
