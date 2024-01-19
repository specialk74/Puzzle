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