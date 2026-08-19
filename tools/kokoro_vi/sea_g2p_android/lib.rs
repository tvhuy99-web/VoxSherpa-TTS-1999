//! Android C FFI adapter for the Apache-2.0 `sea-g2p` Vietnamese phonemizer.
pub mod core;
pub mod g2p;
pub mod lang;
pub mod punc;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
struct Engine { g2p: g2p::G2PEngine }
fn to_c_string(value: String) -> *mut c_char {
    CString::new(value).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}
#[no_mangle]
pub extern "C" fn sea_g2p_create(dictionary_path: *const c_char) -> *mut Engine {
    if dictionary_path.is_null() { return std::ptr::null_mut(); }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let path = unsafe { CStr::from_ptr(dictionary_path) }.to_str().ok()?;
        let engine = g2p::G2PEngine::new(path).ok()?;
        Some(Box::into_raw(Box::new(Engine { g2p: engine })))
    }));
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}
#[no_mangle]
pub extern "C" fn sea_g2p_phonemize(handle: *mut Engine, text: *const c_char) -> *mut c_char {
    if handle.is_null() || text.is_null() { return std::ptr::null_mut(); }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = unsafe { CStr::from_ptr(text) }.to_str().ok()?;
        let engine = unsafe { &(*handle).g2p };
        Some(to_c_string(engine.phonemize(input)))
    }));
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}
#[no_mangle]
pub extern "C" fn sea_g2p_free_string(value: *mut c_char) {
    if !value.is_null() { unsafe { drop(CString::from_raw(value)); } }
}
#[no_mangle]
pub extern "C" fn sea_g2p_destroy(handle: *mut Engine) {
    if !handle.is_null() { unsafe { drop(Box::from_raw(handle)); } }
}
