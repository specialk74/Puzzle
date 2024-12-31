use anyhow::anyhow;
use opencv::core::Point;
use opencv::core::Vector;
use opencv::{self as cv, prelude::*};
use rand::Rng;

pub fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    match cv::imgproc::cvt_color(&phase, &mut new_phase, cv::imgproc::COLOR_BGR2GRAY, 0) {
        Ok(_) => {}
        Err(err) => {
            println!("To Grey Error: {}", err);
            return Err(anyhow!(err));
        }
    }

    Ok(new_phase)
}

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

pub fn show_image(text: &str, img: &Mat) {
    let _ = cv::highgui::imshow(text, img);
}

pub fn find_bounding_rect(
    contours: &Vector<Vector<Point>>,
) -> Result<cv::core::Rect, anyhow::Error> {
    let mut rect = cv::core::Rect::default();

    let mut max_rect = cv::core::Rect::default();
    for contour in contours {
        rect = match cv::imgproc::bounding_rect(&contour) {
            Ok(val) => val,
            Err(err) => {
                println!("find_bounding_rect - err: {}", err);
                return Err(anyhow!(err));
            }
        };
        if rect.width > max_rect.width {
            max_rect = rect;
        }
    }
    Ok(rect)
}

pub fn write_image(name: &str, phase: &Mat) -> Result<bool, opencv::Error> {
    cv::imgcodecs::imwrite(name, &phase, &cv::core::Vector::default())
}

pub fn find_contour(phase: &Mat) -> Result<Vector<Vector<Point>>, anyhow::Error> {
    let mut original_contour_values: Vector<Vector<Point>> = Vector::new();
    let mut contour_values = Vector::new();
    cv::imgproc::find_contours(
        &phase,
        &mut original_contour_values,
        cv::imgproc::RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut biggest = 0;
    // Take only the first contour with len greater than 1000
    for first in &original_contour_values {
        if biggest < first.len() {
            biggest = first.len();
        }
    }

    for first in &original_contour_values {
        if biggest == first.len() {
            contour_values.push(first);
            break;
        }
    }

    Ok(contour_values)
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
    let mut corners = Vector::new();
    let max_corners = 4;
    let quality_level = 0.1;
    let mut distance = 700.0;
    let mut block_size;
    let use_harris_detector: bool = true;
    let k: f64 = 0.1;
    let mut min_corners = Vector::new();
    let mut points: Vector<Point> = Vector::new();
    let mut max_tot_distance = 0.0;
    loop {
        block_size = 60;
        loop {
            match cv::imgproc::good_features_to_track(
                &phase,
                &mut corners,
                max_corners,
                quality_level,
                distance,
                &cv::core::no_array(),
                block_size,
                use_harris_detector,
                k,
            ) {
                Ok(_) => {}
                Err(err) => println!(
                    "Error on find_corners (block_size {}): {} with ",
                    block_size, err
                ),
            };

            if corners.len() == 4 {
                let value_min_area = cv::imgproc::min_area_rect(&corners)?;

                points.clear();
                let min_area_center = Point::new(
                    value_min_area.center.x as i32,
                    value_min_area.center.y as i32,
                );
                let diff = min_area_center - *center;
                points.push(diff);

                let p1_diff = corners.get(0)? - *center;
                points.clear();
                points.push(p1_diff);

                let p2_diff = corners.get(1)? - *center;
                points.clear();
                points.push(p2_diff);

                let p3_diff = corners.get(2)? - *center;
                points.clear();
                points.push(p3_diff);

                let p4_diff = corners.get(3)? - *center;
                points.clear();
                points.push(p4_diff);

                points.push(p1_diff);
                points.push(p2_diff);
                points.push(p3_diff);
                let tot_distance = cv::core::norm_def(&points)?;

                if max_tot_distance < tot_distance {
                    min_corners = corners.clone();
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
