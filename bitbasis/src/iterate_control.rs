pub struct IterControl;

pub fn itercontrol(_: usize, _: &[usize], _: &[usize]) -> IterControl {
    unimplemented!("itercontrol is implemented in Task 2");
}

pub fn group_shift(_: usize, _: &mut [usize]) -> (Vec<usize>, Vec<usize>) {
    unimplemented!("group_shift is implemented in Task 2");
}

pub fn controller(_: &[usize], _: &[usize]) -> impl Fn(usize) -> bool {
    move |_| unimplemented!("controller is implemented in Task 2")
}
