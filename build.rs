fn main() {
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-link-lib=lapack");
}
