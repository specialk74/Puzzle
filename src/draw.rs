use crate::cv_utils::get_color;
use anyhow::anyhow;
use opencv::core::Point;
use opencv::core::Vector;
use opencv::{self as cv, prelude::*};

pub fn draw_four_angles(corners: &Vector<Point>, phase: &mut Mat) -> Result<(), anyhow::Error> {
    for point in corners.iter() {
        match cv::imgproc::circle(
            phase,
            point,
            20,
            cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0),
            cv::imgproc::FILLED,
            cv::imgproc::LINE_8,
            0,
        ) {
            Ok(_) => {}
            Err(err) => {
                println!("Error on draw_four_angles - error: {}", err);
                return Err(anyhow!(err));
            }
        }
    }
    Ok(())
}

pub fn draw_internal_contour(
    contour: &Vector<Vector<Point>>,
    phase: &mut Mat,
) -> Result<(), anyhow::Error> {
    let zero_offset = Point::new(0, 0);
    let thickness: i32 = 20;

    for index in 0..contour.len() {
        match cv::imgproc::draw_contours(
            phase,
            &contour,
            index as i32,
            get_color(),
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset,
        ) {
            Ok(_) => {}
            Err(err) => {
                println!(
                    "Error on draw_internal_contour - index: {} - error: {}",
                    index, err
                );
                return Err(anyhow!(err));
            }
        }
    }
    Ok(())
}
