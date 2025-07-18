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
use sal_core::error::Error;
///
/// This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation.
/// 
/// It was introduced by Andrew B. Godbehere, Akihiro Matsukawa, and Ken Goldberg in their paper
/// "Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation" in 2012.
/// As per the paper, the system ran a successful interactive audio art installation called
/// “Are We There Yet?” from March 31 - July 31 2011 at the Contemporary Jewish Museum in San Francisco, California.
/// 
/// It uses first few (120 by default) frames for background modelling.
/// It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects using Bayesian inference.
/// The estimates are adaptive; newer observations are more heavily weighted than old observations to accommodate variable illumination.
/// Several morphological filtering operations like closing and opening are done to remove unwanted noise.
/// You will get a black window during first few frames.
/// 
/// It would be better to apply morphological opening to the result to remove the noises. 
pub struct RemoveBackground {

}
//
//
impl RemoveBackground {
    pub fn new() -> Self {
        Self {  }
    }
    ///
    /// Performs algoritm
    pub fn eval(&self) -> Result<(), Error> {
        let dbg = "RemoveBackground";
        let error = Error::new(dbg, "eval");
        let paths = std::fs::read_dir("./").unwrap();

        highgui::named_window("Frame", highgui::WINDOW_AUTOSIZE)
            .map_err(|err| error.pass(err.to_string()))?;
        
        for path in paths {
            match path {
                Ok(dir) => {
                    let path = dir.path();
                    if path.is_file() {
                        log::debug!("{dbg}.eval | path: {}", path.display());
                        let mut pattern_img = match imgcodecs::imread(&path.to_str().unwrap(), imgcodecs::IMREAD_COLOR) {
                            Ok(img) => {
                                log::debug!("{dbg}.eval | file read successfully: {:?}", img.size());
                                highgui::imshow("Frame", &img)
                                    .map_err(|err| error.pass(err.to_string()))?;
                                highgui::wait_key(1).unwrap();
                                img
                            },
                            Err(err) => {
                                panic!("{dbg}.eval | can't read file:\n\t{:?}", err);
                            },
                        };
                    }
                }
                Err(err) => log::warn!("{dbg}.eval | IO Error: {:?}", err),
            }
        }        
        Ok(())
    }
}