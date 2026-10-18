//! Own the worker and every COLMAP/FFmpeg descendant as one Windows job.
use std::{io, mem::size_of, ptr};
use windows_sys::Win32::{
    Foundation::CloseHandle,
    System::{
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
        Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE},
    },
};

pub struct WorkerJob {
    handle: usize,
}

impl WorkerJob {
    pub fn attach(pid: u32) -> io::Result<Self> {
        unsafe {
            let handle = CreateJobObjectW(ptr::null(), ptr::null());
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self {
                handle: handle as usize,
            };
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
            if process.is_null() {
                return Err(io::Error::last_os_error());
            }
            let assigned = AssignProcessToJobObject(handle, process);
            let error = if assigned == 0 {
                Some(io::Error::last_os_error())
            } else {
                None
            };
            CloseHandle(process);
            if let Some(error) = error {
                return Err(error);
            }
            Ok(job)
        }
    }
}

impl Drop for WorkerJob {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        os::windows::process::CommandExt,
        process::{Command, Stdio},
        thread,
        time::Duration,
    };
    use windows_sys::Win32::System::{
        JobObjects::{JobObjectBasicProcessIdList, QueryInformationJobObject},
        Threading::{WaitForSingleObject, PROCESS_SYNCHRONIZE},
    };

    #[repr(C)]
    struct JobPids {
        assigned: u32,
        count: u32,
        pids: [usize; 64],
    }

    #[test]
    fn closing_job_terminates_worker_and_descendant() {
        // Assign before triggering the descendant so inheritance has no race.
        let mut child = Command::new("cmd.exe")
            .args([
                "/d",
                "/s",
                "/c",
                "set /p trigger= & ping.exe -n 60 127.0.0.1 >nul",
            ])
            .creation_flags(0x08000000)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let job = WorkerJob::attach(child.id()).unwrap();
        use std::io::Write;
        child.stdin.take().unwrap().write_all(b"start\n").unwrap();
        let mut pids = JobPids {
            assigned: 0,
            count: 0,
            pids: [0; 64],
        };
        for _ in 0..50 {
            let ok = unsafe {
                QueryInformationJobObject(
                    job.handle as _,
                    JobObjectBasicProcessIdList,
                    &mut pids as *mut _ as *mut _,
                    size_of::<JobPids>() as u32,
                    ptr::null_mut(),
                )
            };
            assert_ne!(ok, 0, "Cannot inspect worker job");
            if pids.count >= 2 {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(pids.count >= 2, "The probe must contain a real descendant");
        let handles: Vec<_> = pids.pids[..pids.count as usize]
            .iter()
            .map(|pid| unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, *pid as u32) })
            .collect();
        assert!(handles.iter().all(|handle| !handle.is_null()));
        drop(job);
        for handle in handles {
            assert_eq!(
                unsafe { WaitForSingleObject(handle, 3000) },
                0,
                "Orphan process survived job closure"
            );
            unsafe {
                CloseHandle(handle);
            }
        }
        // Windows chooses the exit code for KILL_ON_JOB_CLOSE; termination is
        // established by the signaled handles for both processes above.
        child.wait().unwrap();
    }
}
