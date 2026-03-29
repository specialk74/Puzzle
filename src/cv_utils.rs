use anyhow::anyhow;
use opencv::core::AlgorithmHint;
use opencv::core::Point;
use opencv::core::Vector;
use opencv::{self as cv, prelude::*};
use rand::Rng;

pub fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    match cv::imgproc::cvt_color(
        &phase,
        &mut new_phase,
        cv::imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    ) {
        Ok(_) => {}
        Err(err) => {
            println!("To Grey Error: {}", err);
            return Err(anyhow!(err));
        }
    }

    Ok(new_phase)
}

#[allow(dead_code)]
pub fn wait_key(delay: i32) -> Result<i32, cv::Error> {
    cv::highgui::wait_key(delay)
}

pub fn get_black_color() -> cv::core::Scalar {
    cv::core::Scalar::new(0.0, 0.0, 0.0, 255.0)
}

pub fn get_white_color() -> cv::core::Scalar {
    cv::core::Scalar::new(255.0, 255.0, 255.0, 255.0)
}

pub fn blur(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let ksize = cv::core::Size::new(15, 15);
    let mut new_phase = cv::core::Mat::default();

    match cv::imgproc::blur_def(&phase, &mut new_phase, ksize) {
        Ok(()) => {}
        Err(err) => {
            println!("Blur Err: {:?}", err);
            return Err(anyhow!(err));
        }
    }
    Ok(new_phase)
}

pub fn get_color() -> cv::core::Scalar {
    let mut rng = rand::thread_rng();
    let n1 = rng.gen_range(0.0..255.0);
    let n2 = rng.gen_range(0.0..255.0);
    let n3 = rng.gen_range(0.0..255.0);
    cv::core::Scalar::new(n1, n2, n3, 255.0)
}

#[allow(dead_code)]
pub fn show_image(text: &str, img: &Mat) {
    let _ = cv::highgui::imshow(text, img);
}

#[allow(dead_code)]
pub fn find_bounding_rect(
    contours: &Vector<Vector<Point>>,
) -> Result<cv::core::Rect, anyhow::Error> {
    let mut max_rect = cv::core::Rect::default();
    for contour in contours {
        let rect = cv::imgproc::bounding_rect(&contour)?;
        if rect.width > max_rect.width {
            max_rect = rect;
        }
    }
    Ok(max_rect)
}

pub fn write_image(name: &str, phase: &Mat) -> Result<bool, opencv::Error> {
    cv::imgcodecs::imwrite(name, &phase, &cv::core::Vector::default())
}

pub fn find_contour(phase: &Mat) -> Result<Vector<Vector<Point>>, anyhow::Error> {
    let mut all: Vector<Vector<Point>> = Vector::new();
    cv::imgproc::find_contours(
        phase,
        &mut all,
        cv::imgproc::RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut result = Vector::new();
    let mut biggest_len = 0;
    let mut biggest = Vector::new();
    for c in &all {
        if c.len() > biggest_len {
            biggest_len = c.len();
            biggest = c;
        }
    }
    if biggest_len > 0 {
        result.push(biggest);
    }
    Ok(result)
}

pub fn morph(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    let anchor = Point::new(-1, -1);
    let ksize = cv::core::Size::new(5, 5);
    let kernel = cv::imgproc::get_structuring_element(0, ksize, anchor)?;
    cv::imgproc::morphology_ex(
        &phase,
        &mut new_phase,
        cv::imgproc::MORPH_OPEN,
        &kernel,
        anchor,
        1,
        cv::core::BORDER_CONSTANT,
        cv::imgproc::morphology_default_border_value()?,
    )?;

    Ok(new_phase)
}

pub fn bitwise(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::core::bitwise_not_def(&phase, &mut new_phase)?;

    Ok(new_phase)
}

pub fn threshold(phase: &Mat, threshold_value: i32) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::imgproc::threshold(
        &phase,
        &mut new_phase,
        threshold_value as f64,
        255.0,
        cv::imgproc::THRESH_BINARY,
    )?;

    Ok(new_phase)
}

pub fn read_image(file_name: &str) -> Result<Mat, opencv::Error> {
    cv::imgcodecs::imread(file_name, cv::imgcodecs::IMREAD_COLOR)
}

pub fn find_corners(center: &Point, phase: &Mat) -> Result<(f64, Vector<Point>), anyhow::Error> {
    let max_corners = 4;
    let quality_level = 0.1;
    let use_harris_detector = true;
    let k = 0.1;
    let mut min_corners = Vector::new();
    let mut max_tot_distance = 0.0;
    let mut distance = 700.0;

    loop {
        let mut block_size = 60;
        loop {
            let mut corners = Vector::new();
            if let Err(err) = cv::imgproc::good_features_to_track(
                phase,
                &mut corners,
                max_corners,
                quality_level,
                distance,
                &cv::core::no_array(),
                block_size,
                use_harris_detector,
                k,
            ) {
                println!("Error on find_corners (block_size {}): {}", block_size, err);
            }

            if corners.len() == 4 {
                let mut points: Vector<Point> = Vector::new();
                points.push(corners.get(0)? - *center);
                points.push(corners.get(1)? - *center);
                points.push(corners.get(2)? - *center);
                points.push(corners.get(3)? - *center);
                let tot_distance = cv::core::norm_def(&points)?;

                if tot_distance > max_tot_distance {
                    min_corners = corners;
                    max_tot_distance = tot_distance;
                }
            }

            block_size += 20;
            if block_size > 90 {
                break;
            }
        }
        distance += 20.0;
        if distance > 1000.0 {
            break;
        }
    }

    Ok((max_tot_distance, min_corners))
}

pub fn get_first_contour(contours: &Vector<Vector<Point>>) -> Result<Vector<Point>, anyhow::Error> {
    contours.get(0).map_err(|err| anyhow!(err))
}

pub fn find_centroid(contours: &Vector<Vector<Point>>) -> Result<Point, anyhow::Error> {
    let first = get_first_contour(contours)?;

    match cv::imgproc::moments_def(&first) {
        Ok(moment) => {
            let cx = moment.m10 / moment.m00;
            let cy = moment.m01 / moment.m00;
            Ok(Point::new(cx as i32, cy as i32))
        }
        Err(err) => Err(anyhow!(err)),
    }
}

pub fn sub_process(
    grey_phase: &Mat,
    threshold_value: i32,
) -> Result<Vector<Vector<Point>>, anyhow::Error> {
    let im = threshold(grey_phase, threshold_value)?;
    let im = bitwise(&im)?;
    let im = morph(&im)?;
    find_contour(&im)
}

pub fn find_limit(vector_traslated: &Vector<Point>) -> (i32, i32, i32, i32) {
    let mut x_max = 0;
    let mut y_min = i32::MAX;
    let mut y_max = 0;
    let mut x_min = i32::MAX;

    for point in vector_traslated.iter() {
        if point.x < x_min {
            x_min = point.x;
        }
        if point.x > x_max {
            x_max = point.x;
        }
        if point.y < y_min {
            y_min = point.y;
        }
        if point.y > y_max {
            y_max = point.y;
        }
    }
    (x_max, y_max, x_min, y_min)
}
