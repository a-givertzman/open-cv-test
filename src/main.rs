#![allow(non_snake_case)]

use opencv::{
    core,
    types,
    objdetect,
    highgui,
    imgcodecs,
    imgproc,
    videoio,
    Result,
    prelude::*,
    features2d,
};

fn main() -> Result<()> {
    // faceDetection()?;
    // return Ok(());
    let mut patternImg = match imgcodecs::imread("./assets/patterns/p1.png", imgcodecs::IMREAD_COLOR) {
        Ok(img) => {
            println!("[main] file read successfully:\n\t{:?}", img.size());
            img
        },
        Err(err) => {
            panic!("[main] can't read file:\n\t{:?}", err);
        },
    };
    match videoio::VideoCapture::new(0, videoio::CAP_ANY) {
        Ok(mut cam) => {
            cam.set(videoio::CAP_PROP_FPS, 10f64)?;
            let mut frame = Mat::default();
            loop {
                cam.read(&mut frame)?;
                let mut dstImg = core::Mat::default();
                // imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                // let mut faces = types::VectorOfRect::new();
                // if faces.len() > 0 {
                //     println!("faces: {:?}", faces);
                // }
                // for face in faces.into_iter() {
                //     imgproc::rectangle(
                //         &mut img, 
                //         face, 
                //         core::Scalar::new(0f64, 255f64, 0f64, 0f64), 
                //         2, 
                //         imgproc::LINE_8, 
                //         0,
                //     )?;
                // }
                orbMatch(&mut patternImg, &mut dstImg, 0.7)?;

                highgui::named_window("img", highgui::WINDOW_AUTOSIZE)?;
                // highgui::imshow("img", &patternImg)?;
                // highgui::imshow("dst img", &dstImg)?;
                highgui::imshow("frame", &frame)?;
                match highgui::wait_key(1) {
                    Ok(key) => {
                        if key == 'q' as i32 {
                            break;
                        }
                    },
                    Err(err) => panic!("highgui::wait_key error: {:?}", err),
                };
            }        


        },
        Err(err) => {
            println!("[main] getting videoCapture error:\n\t{:?}", err);
        },
    };
    Ok(())
}


fn orbMatch(patternImg: &mut Mat, dstImg: &mut Mat, matchRatio: f32) -> Result<()>{
    let orb = opencv::features2d::ORB::create(
        500,
        1.2,
        8,
        31,
        0,
        2,
        features2d::ORB_ScoreType::HARRIS_SCORE,
        31,
        20,        
        // nfeatures, scale_factor, nlevels, edge_threshold, first_level, wta_k, score_type, patch_size, fast_threshold
    );
    match orb {
        Ok(mut orb) => {
            let mut keypointsPattern = core::Vector::default();
            let mut keypointsDstImg = core::Vector::default();
            let mut descPattern = core::Mat::default();
            let mut descDstImg = core::Mat::default();
        
            match orb.detect_and_compute(patternImg, &core::no_array(), &mut keypointsPattern, &mut descPattern, false) {
                Ok(_) => {},
                Err(err) => {
                    println!("[orbMatch] detect_and_compute error:\n\t{:?}", err);
                },
            };
            match orb.detect_and_compute(dstImg, &core::no_array(), &mut keypointsDstImg, &mut descDstImg, false) {
                Ok(_) => {},
                Err(err) => {
                    println!("[orbMatch] detect_and_compute error:\n\t{:?}", err);
                },
            };
            let mut bfMatches: core::Vector<core::Vector<core::DMatch>> = core::Vector::default();    // core::Vector<core::Vector<core::DMatch>>
            match opencv::features2d::DescriptorMatcher::create("FlannBased") {
                Ok(mut bf) => {
                    let masks = core::Mat::default();
                    bf.knn_match(&descPattern, &mut bfMatches, 2, &core::no_array(), false)?;
                    bf
                },
                Err(err) => panic!("[orbMatch] BFMatcher create error: {:?}", err),
            };
            let mut goodMatches = core::Vector::default();
            for mm in bfMatches {
                let m0 = mm.get(0)?;
                let m1 = mm.get(1)?;
                if m0.distance < matchRatio * m1.distance {
                    goodMatches.push(m0);
                }
            }
            println!("[orbMatch] matches: {:?}", goodMatches);
            opencv::features2d::draw_matches(
                patternImg, 
                &keypointsPattern, 
                &dstImg.clone(),
                &keypointsDstImg, 
                &goodMatches, 
                dstImg, 
                core::Scalar::new(0f64, 255f64, 0f64, 0f64), 
                core::Scalar::new(0f64, 255f64, 0f64, 0f64), 
                &core::Vector::default(), 
                features2d::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,
            ).unwrap();
            // features2d::draw_keypoints(
            //     imgPattern,
            //     &keypoints,
            //     dstImg,
            //     core::VecN([0., 255., 0., 255.]),
            //     features2d::DrawMatchesFlags::DEFAULT,
            // )?;
            // imgproc::rectangle(
            //     dstImg,
            //     core::Rect::from_points(core::Point::new(0, 0), core::Point::new(50, 50)),
            //     core::VecN([255., 0., 0., 0.]),
            //     -1,
            //     imgproc::LINE_8,
            //     0,
            // )?;
            // // Use SIFT
            // let mut sift = features2d::SIFT::create(0, 3, 0.04, 10., 1.6)?;
            // let mut sift_keypoints = core::Vector::default();
            // let mut sift_desc = core::Mat::default();
            // sift.detect_and_compute(imgPattern, &mask, &mut sift_keypoints, &mut sift_desc, false)?;
            // features2d::draw_keypoints(
            //     &dstImg.clone(),
            //     &sift_keypoints,
            //     dstImg,
            //     core::VecN([0., 0., 255., 255.]),
            //     features2d::DrawMatchesFlags::DEFAULT,
            // )?;

        },
        Err(err) => {
            println!("[orbMatch] creating ORB error:\n\t{:?}", err);
        },
    };
    Ok(())
}


fn faceDetection() -> Result<()> {
    let path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    let videoCapture = videoio::VideoCapture::new(0, videoio::CAP_ANY);
    match videoCapture {
        Ok(mut cam) => {
            let mut img = Mat::default();
            let mut faceDetector = objdetect::CascadeClassifier::new(path)?;
            cam.set(videoio::CAP_PROP_FPS, 10f64)?;
            loop {
                cam.read(&mut img)?;
                let mut gray = Mat::default();
                imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                let mut faces = opencv::core::Vector::<opencv::core::Rect>::new();
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