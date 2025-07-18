mod remove_background;
use debugging::session::debug_session::{Backtrace, DebugSession, LogLevel};
use crate::remove_background::RemoveBackground;

fn main() {
    DebugSession::init(LogLevel::Debug, Backtrace::Short);
    let remove_background = RemoveBackground::new();
    remove_background.eval().unwrap();
}
