#![allow(non_snake_case)]
use opencv::{
    core,
    types,
    objdetect,
    highgui,
    imgproc,
    videoio,
    Result,
    prelude::*,
};

fn main() -> Result<()> {
    let path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    let videoCapture = videoio::VideoCapture::new(0, videoio::CAP_ANY);
    match videoCapture {
        Ok(mut cam) => {
            let mut img = Mat::default();
            let mut faceDetector = objdetect::CascadeClassifier::new(path)?;
            loop {
                cam.read(&mut img)?;
                let mut gray = Mat::default();
                imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                let mut faces = types::VectorOfRect::new();
                faceDetector.detect_multi_scale(
                    &gray, 
                    &mut faces, 
                    1.1, 
                    10, 
                    objdetect::CASCADE_SCALE_IMAGE, 
                    core::Size::new(50,50), 
                    core::Size::new(1000, 1000)
                )?;
                if faces.len() > 0 {
                    println!("faces: {:?}", faces);
                }
                for face in faces.into_iter() {
                    imgproc::rectangle(
                        &mut img, 
                        face, 
                        core::Scalar::new(0f64, 255f64, 0f64, 0f64), 
                        2, 
                        imgproc::LINE_8, 
                        0,
                    )?;
                }
                highgui::named_window("img", highgui::WINDOW_AUTOSIZE)?;
                highgui::imshow("img", &img)?;
                highgui::wait_key(1)?;
            }        
        },
        Err(err) => {
            println!("error geting videoCapture: {:?}", err);
        },
    };
    Ok(())
}
