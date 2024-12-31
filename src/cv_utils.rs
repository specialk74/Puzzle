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
    _original_image: &Mat,
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
