use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    env, io,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tauri::{AppHandle, Emitter, Manager};
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{oneshot, Mutex as AsyncMutex},
};

type Reply = Result<Value, String>;
type Pending = HashMap<String, oneshot::Sender<Reply>>;
const MAX_REQUESTS: usize = 128;
const MAX_FRAME: usize = 4 * 1024 * 1024;
const WORKER_BOOTSTRAP: &str = "import sys; sys.dont_write_bytecode=True; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; from local_postshot.worker import main; sys.exit(main())";

#[derive(Default)]
pub struct EngineSupervisor {
    worker: AsyncMutex<Option<Arc<Worker>>>,
    next_id: AtomicU64,
    closing: AtomicBool,
}

struct Worker {
    stdin: AsyncMutex<Option<ChildStdin>>,
    child: AsyncMutex<Child>,
    pending: Mutex<Pending>,
    alive: AtomicBool,
    #[cfg(windows)]
    job: Mutex<Option<crate::native_job::WorkerJob>>,
}

#[derive(Deserialize)]
struct Runtime {
    python: PathBuf,
    #[serde(default)]
    engine_root: Option<PathBuf>,
    #[serde(default)]
    pipeline_root: Option<PathBuf>,
}

fn allowed_method(method: &str, params: &Value) -> bool {
    match method {
        "studio.catalog" => true,
        "studio.operation" => matches!(
            params["action"].as_str(),
            Some(
                "control_run"
                    | "run_stage"
                    | "run_status"
                    | "experiment.read"
                    | "create_location"
                    | "create_scene"
                    | "create_capture"
                    | "capture_config"
                    | "normalize_capture"
                    | "update_hierarchy"
                    | "delete_hierarchy"
                    | "source_hashes"
                    | "relink_capture"
                    | "probe_capture"
                    | "ingest"
                    | "create_run"
                    | "resume_run"
                    | "run_config"
                    | "masks.review"
                    | "masks.read"
                    | "masks.commit"
                    | "finalize_masks"
                    | "qa_report"
                    | "review_qa"
                    | "segment_choices"
                    | "experiments"
                    | "train_segment"
                    | "training_options"
                    | "control_training"
                    | "training_state"
                    | "sources.list"
                    | "preview.request"
                    | "preview.read"
                    | "cleanup"
                    | "verified_models"
                    | "model_open"
                    | "inspect_external_import"
                    | "prepare_external_import"
                    | "train_external_import"
                    | "list_external_imports"
                    | "relink_external_import"
                    | "select_sharp"
            )
        ),
        "editor.operation" => matches!(
            params["action"].as_str(),
            Some(
                "state"
                    | "select_rect"
                    | "invert"
                    | "clear"
                    | "delete"
                    | "preview"
                    | "commit"
                    | "cancel_preview"
                    | "undo"
                    | "redo"
                    | "render"
            )
        ),
        "health" | "status" | "project.list" | "project.create" | "project.open"
        | "project.update" | "project.import" | "reconstruct" | "train" | "pause" | "cancel"
        | "resume" | "suspend" | "unpause" | "checkpoint" | "stop" | "model.import"
        | "model.export" | "model.clean" | "render" | "evaluate" => true,
        _ => false,
    }
}

impl EngineSupervisor {
    pub async fn call(&self, app: &AppHandle, method: String, params: Value) -> Reply {
        if !allowed_method(&method, &params) {
            return Err(format!("不支持的引擎方法：{method}"));
        }
        if !params.is_object() {
            return Err("引擎参数必须为 JSON 对象".into());
        }
        // Startup is serialized; a request never keeps this lock while awaiting its reply.
        let worker = {
            let mut slot = self.worker.lock().await;
            if self.closing.load(Ordering::Acquire) {
                return Err("应用正在关闭".into());
            }
            if slot
                .as_ref()
                .is_some_and(|w| !w.alive.load(Ordering::Acquire))
            {
                if let Some(old) = slot.take() {
                    old.stop().await;
                }
            }
            if slot.is_none() {
                *slot = Some(Worker::spawn(app.clone()).await?);
            }
            slot.as_ref().unwrap().clone()
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let mut frame = serde_json::to_vec(&json!({"id": id, "method": method, "params": params}))
            .map_err(|e| e.to_string())?;
        if frame.len() > MAX_FRAME {
            return Err("请求超过 4 MiB 限制".into());
        }
        frame.push(b'\n');
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = worker.pending.lock().map_err(|_| "引擎请求锁不可用")?;
            if !worker.alive.load(Ordering::Acquire) {
                return Err("引擎进程已退出，请重试".into());
            }
            if pending.len() >= MAX_REQUESTS {
                return Err("等待中的引擎请求过多".into());
            }
            pending.insert(id.clone(), tx);
        }
        let written = {
            let mut slot = worker.stdin.lock().await;
            match slot.as_mut() {
                Some(stdin) => match stdin.write_all(&frame).await {
                    Ok(()) => stdin.flush().await,
                    Err(e) => Err(e),
                },
                None => Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "worker stdin is closed",
                )),
            }
        };
        if let Err(e) = written {
            worker.fail_all(format!("引擎写入失败：{e}"));
            worker.stop().await;
        }
        // Training can take hours. Short management calls get a bounded response window.
        let result = if matches!(
            method.as_str(),
            "health"
                | "status"
                | "project.list"
                | "project.open"
                | "project.update"
                | "studio.catalog"
        ) {
            match tokio::time::timeout(Duration::from_secs(45), rx).await {
                Ok(result) => result.map_err(|_| "引擎连接已关闭".to_string())?,
                Err(_) => {
                    if let Ok(mut pending) = worker.pending.lock() {
                        pending.remove(&id);
                    }
                    return Err(format!("{method} 请求 45 秒未响应；可重试，或取消当前任务"));
                }
            }
        } else {
            rx.await.map_err(|_| "引擎连接已关闭".to_string())?
        };
        result
    }

    pub async fn shutdown(&self) {
        self.closing.store(true, Ordering::Release);
        if let Some(worker) = self.worker.lock().await.take() {
            worker.shutdown_gracefully().await;
        }
    }
}

impl Worker {
    async fn spawn(app: AppHandle) -> Result<Arc<Self>, String> {
        let runtime_path = runtime_config_path()?;
        let runtime_text = tokio::task::spawn_blocking({
            let path = runtime_path.clone();
            move || std::fs::read_to_string(path)
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| {
            format!(
                "无法读取运行环境 {}：{e}。请配置 GSSTUDIO_RUNTIME。",
                runtime_path.display()
            )
        })?;
        let mut runtime: Runtime =
            serde_json::from_str(runtime_text.trim_start_matches('\u{feff}'))
                .map_err(|e| format!("运行环境 JSON 无效（{}）：{e}", runtime_path.display()))?;
        if let Some(override_path) = runtime_override_path()? {
            let text = std::fs::read_to_string(&override_path).map_err(|e| e.to_string())?;
            let fields: Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))
                .map_err(|e| e.to_string())?;
            if let Some(value) = fields.get("python").and_then(Value::as_str) {
                runtime.python = resolve_relative(&override_path, Path::new(value))?;
            }
            if let Some(value) = fields.get("engine_root").and_then(Value::as_str) {
                runtime.engine_root = Some(resolve_relative(&override_path, Path::new(value))?);
            }
            if let Some(value) = fields.get("pipeline_root").and_then(Value::as_str) {
                runtime.pipeline_root = Some(resolve_relative(&override_path, Path::new(value))?);
            }
        }
        if !runtime.python.is_absolute() {
            runtime.python = runtime_path
                .parent()
                .ok_or("无法解析运行环境目录")?
                .join(&runtime.python);
        }
        if !runtime.python.is_absolute() || !runtime.python.is_file() {
            return Err(format!(
                "运行环境 python 必须指向存在的绝对路径：{}",
                runtime.python.display()
            ));
        }
        let engine_dir = module_dir(runtime.engine_root.as_deref(), &runtime_path, "engine")?;
        let pipeline_dir = module_dir(runtime.pipeline_root.as_deref(), &runtime_path, "pipeline")?;
        if !engine_dir.join("local_postshot/worker.py").is_file() {
            return Err(format!(
                "缺少引擎模块：{}。请配置 runtime.json engine_root。",
                engine_dir.display()
            ));
        }
        if !pipeline_dir
            .join("gsstudio/interfaces/worker/main.py")
            .is_file()
        {
            return Err(format!(
                "缺少 GS Studio 模块：{}。请配置 runtime.json pipeline_root。",
                pipeline_dir.display()
            ));
        }
        let mut command = Command::new(&runtime.python);
        command.env_clear();
        // No inherited Python, CUDA, proxy, credentials or donor virtualenv state.
        for key in [
            "SystemRoot",
            "WINDIR",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "LOCALAPPDATA",
            "APPDATA",
            "PROGRAMDATA",
            "COMSPEC",
            "NUMBER_OF_PROCESSORS",
            // Explicit user configuration survives the otherwise clean environment.
            "GSSTUDIO_RUNTIME_OVERRIDE",
            "GSSTUDIO_SETTINGS_DIR",
            "GSSTUDIO_DATA_ROOT",
            "GSSTUDIO_GPU_LOCK_PATH",
            "GSSTUDIO_CACHE_ROOT",
            "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR",
            "CUDA_PATH",
            "CUDA_HOME",
            "GSSTUDIO_MEDIA_HELPER",
            "INSTA360_MEDIA_SDK_ROOT",
            "GSSTUDIO_REALITYSCAN_CLI",
            "GSSTUDIO_REALITYSCAN_EXPORT_PARAMS",
        ] {
            if let Some(value) = env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut path = vec![runtime
            .python
            .parent()
            .ok_or("Python 路径没有父目录")?
            .to_path_buf()];
        if let Some(windows) = env::var_os("SystemRoot") {
            path.push(PathBuf::from(windows).join("System32"));
        }
        command
            // Explicit sys.path also works with portable Python's isolated ._pth.
            .args(["-I", "-B", "-u", "-c", WORKER_BOOTSTRAP])
            .arg(&engine_dir)
            .arg(&pipeline_dir)
            .current_dir(&engine_dir)
            .env("PATH", env::join_paths(path).map_err(|e| e.to_string())?)
            .env(
                "PYTHONPATH",
                env::join_paths([&engine_dir, &pipeline_dir]).map_err(|e| e.to_string())?,
            )
            .env("PYTHONNOUSERSITE", "1")
            .env_remove("PYTHONHOME")
            .env_remove("VIRTUAL_ENV")
            .env("PYTHONIOENCODING", "utf-8")
            .env("GSSTUDIO_RUNTIME", &runtime_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Python's pure configuration resolver owns tokens, saved settings and
        // user overrides for cache, projects, logs and the ONE shared GPU lock.
        #[cfg(windows)]
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW; never requires Python on PATH.
        let mut child = command
            .spawn()
            .map_err(|e| format!("无法启动 Python {}：{e}", runtime.python.display()))?;
        #[cfg(windows)]
        let job = match crate::native_job::WorkerJob::attach(
            child.id().ok_or("无法读取 worker 进程 ID")?,
        ) {
            Ok(job) => job,
            Err(error) => {
                let _ = child.kill().await;
                return Err(format!("无法隔离 worker 进程树：{error}"));
            }
        };
        let stdin = child.stdin.take().ok_or("无法打开引擎 stdin")?;
        let stdout = child.stdout.take().ok_or("无法打开引擎 stdout")?;
        let stderr = child.stderr.take().ok_or("无法打开引擎 stderr")?;
        let worker = Arc::new(Self {
            stdin: AsyncMutex::new(Some(stdin)),
            child: AsyncMutex::new(child),
            pending: Mutex::new(HashMap::new()),
            alive: AtomicBool::new(true),
            #[cfg(windows)]
            job: Mutex::new(Some(job)),
        });

        let reader_worker = worker.clone();
        let reader_app = app.clone();
        tauri::async_runtime::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                let line = match bounded_line(&mut reader, MAX_FRAME, false).await {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        log(&reader_app, "error", format!("引擎协议读取失败：{e}"));
                        break;
                    }
                };
                if line.trim().is_empty() {
                    continue;
                }
                let message: Value = match serde_json::from_str(&line) {
                    Ok(message) => message,
                    Err(e) => {
                        log(&reader_app, "error", format!("引擎返回无效 JSON：{e}"));
                        break;
                    }
                };
                match message.get("type").and_then(Value::as_str) {
                    Some("event") => {
                        admit_assets(&reader_app, &message);
                        let _ = reader_app.emit("engine-event", &message);
                    }
                    Some("result" | "error") => {
                        if let Some(id) = message.get("id").and_then(Value::as_str) {
                            let sender = reader_worker
                                .pending
                                .lock()
                                .ok()
                                .and_then(|mut p| p.remove(id));
                            if let Some(sender) = sender {
                                let result = if message["type"] == "result" {
                                    let result =
                                        message.get("result").cloned().unwrap_or(Value::Null);
                                    admit_assets(&reader_app, &result);
                                    Ok(result)
                                } else {
                                    Err(format_error(&message["error"]))
                                };
                                let _ = sender.send(result);
                            }
                        }
                    }
                    _ => log(&reader_app, "warn", "忽略未知引擎消息类型".into()),
                }
            }
            reader_worker.fail_all("引擎输出连接已关闭，请重试以重新启动引擎".into());
            reader_worker.stop().await;
        });
        let stderr_app = app.clone();
        tauri::async_runtime::spawn(async move {
            let mut reader = BufReader::new(stderr);
            loop {
                match bounded_line(&mut reader, 64 * 1024, true).await {
                    Ok(Some(line)) => log(&stderr_app, "info", line),
                    Ok(None) => break,
                    Err(e) => {
                        log(&stderr_app, "error", format!("引擎 stderr 读取中断：{e}"));
                        break;
                    }
                }
            }
        });
        let monitor = worker.clone();
        tauri::async_runtime::spawn(async move {
            while monitor.alive.load(Ordering::Acquire) {
                let status = monitor.child.lock().await.try_wait();
                match status {
                    Ok(Some(status)) => {
                        let message = format!("引擎进程已退出（{status}），下一次请求将重新启动");
                        monitor.fail_all(message.clone());
                        log(&app, "error", message);
                        monitor.stop().await;
                        break;
                    }
                    Err(e) => {
                        monitor.fail_all(format!("无法读取引擎进程状态：{e}"));
                        monitor.stop().await;
                        break;
                    }
                    _ => {}
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        });
        Ok(worker)
    }

    fn fail_all(&self, message: String) {
        if let Ok(mut pending) = self.pending.lock() {
            self.alive.store(false, Ordering::Release);
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(message.clone()));
            }
        }
    }

    async fn stop(&self) {
        self.fail_all("引擎进程已停止".into());
        // Closing the job kills the full Python/COLMAP/FFmpeg process tree.
        #[cfg(windows)]
        if let Ok(mut job) = self.job.lock() {
            job.take();
        }
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    }

    async fn shutdown_gracefully(&self) {
        // The engine handles EOF by setting pause/control and joining its task,
        // saving a complete training checkpoint after the current iteration.
        self.stdin.lock().await.take();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            if matches!(self.child.lock().await.try_wait(), Ok(Some(_))) {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        self.stop().await;
    }
}

fn runtime_config_path() -> Result<PathBuf, String> {
    let candidate = if let Some(path) = env::var_os("GSSTUDIO_RUNTIME")
        .or_else(|| env::var_os("LOCAL_POSTSHOT_RUNTIME"))
        .filter(|path| !path.is_empty())
    {
        PathBuf::from(path)
    } else {
        let adjacent = env::current_exe()
            .map_err(|e| e.to_string())?
            .parent()
            .ok_or("无法解析程序目录")?
            .join("runtime/runtime.json");
        adjacent
    };
    if candidate.is_absolute() {
        Ok(candidate)
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(candidate))
            .map_err(|e| e.to_string())
    }
}

fn runtime_override_path() -> Result<Option<PathBuf>, String> {
    let explicit = env::var_os("GSSTUDIO_RUNTIME_OVERRIDE").filter(|v| !v.is_empty());
    let path = explicit.as_ref().map(PathBuf::from).or_else(|| {
        env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("GSStudio/runtime.override.json"))
    });
    match path {
        Some(path) if path.is_file() => Ok(Some(path)),
        Some(path) if explicit.is_some() => Err(format!(
            "Configured runtime override is missing: {}",
            path.display()
        )),
        _ => Ok(None),
    }
}

fn resolve_relative(config: &Path, path: &Path) -> Result<PathBuf, String> {
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        config.parent().ok_or("运行环境配置没有父目录")?.join(path)
    })
}

fn module_dir(configured: Option<&Path>, config: &Path, kind: &str) -> Result<PathBuf, String> {
    if let Some(path) = configured {
        return resolve_relative(config, path);
    }
    let adjacent = env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or("无法解析程序目录")?
        .join(kind);
    if adjacent.is_dir() {
        return Ok(adjacent);
    }
    #[cfg(debug_assertions)]
    {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("无法解析开发目录")?;
        Ok(root.join(if kind == "pipeline" { "src" } else { kind }))
    }
    #[cfg(not(debug_assertions))]
    Err(format!(
        "发布包缺少 {kind}；必须提供随包模块目录或 runtime.json 显式路径"
    ))
}

fn format_error(error: &Value) -> String {
    let code = error["code"].as_str().unwrap_or("ENGINE_ERROR");
    let message = error["message"].as_str().unwrap_or("未知引擎错误");
    match error
        .get("details")
        .filter(|d| !d.is_null() && **d != json!({}))
    {
        Some(details) => format!("[{code}] {message}\n{details}"),
        None => format!("[{code}] {message}"),
    }
}

fn log(app: &AppHandle, level: &str, message: String) {
    let _ = app.emit(
        "engine-event",
        json!({"type":"event", "id":"supervisor", "event":"log",
        "data":{"level":level, "message":message}}),
    );
}

// Only engine-returned model assets are admitted. No broad disk glob or remote web origins.
fn admit_assets(app: &AppHandle, value: &Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if matches!(
                    key.as_str(),
                    "path"
                        | "preview_path"
                        | "file"
                        | "source_path"
                        | "image_path"
                        | "mask_path"
                        | "render_path"
                        | "selection_preview"
                        | "native_render"
                ) {
                    if let Some(path) = value.as_str() {
                        let path = Path::new(path);
                        let ext = path
                            .extension()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_ascii_lowercase();
                        if path.is_absolute()
                            && matches!(
                                ext.as_str(),
                                "ply" | "spz" | "png" | "jpg" | "jpeg" | "webp" | "bmp"
                            )
                            && path.is_file()
                        {
                            let _ = app.asset_protocol_scope().allow_file(path);
                        }
                    }
                }
                admit_assets(app, value);
            }
        }
        Value::Array(values) => {
            for value in values {
                admit_assets(app, value);
            }
        }
        _ => {}
    }
}

async fn bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    limit: usize,
    truncate: bool,
) -> io::Result<Option<String>> {
    let mut bytes = Vec::new();
    let mut truncated = false;
    loop {
        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            return if bytes.is_empty() {
                Ok(None)
            } else {
                decode_line(bytes, truncate, truncated).map(Some)
            };
        }
        let end = chunk.iter().position(|b| *b == b'\n');
        let length = end.map_or(chunk.len(), |i| i + 1);
        if bytes.len() + length > limit {
            if !truncate {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "JSONL line exceeds buffer limit",
                ));
            }
            truncated = true;
        }
        let keep = length.min(limit.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk[..keep]);
        reader.consume(length);
        if end.is_some() {
            return decode_line(bytes, truncate, truncated).map(Some);
        }
    }
}

fn decode_line(bytes: Vec<u8>, lossy: bool, truncated: bool) -> io::Result<String> {
    if lossy {
        let mut line = String::from_utf8_lossy(&bytes).into_owned();
        if truncated {
            line.push_str(" … [stderr line truncated at 64 KiB]");
        }
        Ok(line)
    } else {
        String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_explicit_rpc_methods_and_actions_are_allowed() {
        assert!(allowed_method("studio.catalog", &json!({})));
        assert!(allowed_method(
            "studio.operation",
            &json!({"action":"masks.commit"})
        ));
        assert!(allowed_method(
            "editor.operation",
            &json!({"action":"select_rect"})
        ));
        assert!(allowed_method("suspend", &json!({})));
        assert!(allowed_method(
            "studio.operation",
            &json!({"action":"training_options"})
        ));
        for method in ["shell", "exec", "eval", "open_url", "fs.read"] {
            assert!(!allowed_method(method, &json!({})));
        }
        assert!(!allowed_method(
            "studio.operation",
            &json!({"action":"shell"})
        ));
        assert!(!allowed_method(
            "editor.operation",
            &json!({"action":"eval"})
        ));
    }

    #[test]
    fn packaged_config_paths_resolve_against_config_not_working_directory() {
        let config = Path::new("E:/AI/GSStudio/app/runtime/runtime.json");
        assert_eq!(
            resolve_relative(config, Path::new("core-cu129/python.exe")).unwrap(),
            config.parent().unwrap().join("core-cu129/python.exe")
        );
        assert_eq!(
            module_dir(Some(Path::new("../engine")), config, "engine").unwrap(),
            config.parent().unwrap().join("../engine")
        );
    }

    #[cfg(windows)]
    #[test]
    fn actual_worker_bootstrap_consumes_module_paths_before_argparse() {
        // Same bootstrap used by production. A real status request exercises
        // argparse and native-runtime imports without asking for CUDA or jobs.
        let config = runtime_config_path().unwrap();
        let text = std::fs::read_to_string(&config)
            .expect("configured core runtime required for startup regression");
        let runtime: Runtime = serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap();
        let python = resolve_relative(&config, &runtime.python).unwrap();
        let engine = module_dir(runtime.engine_root.as_deref(), &config, "engine").unwrap();
        let pipeline = module_dir(runtime.pipeline_root.as_deref(), &config, "pipeline").unwrap();
        let mut command = std::process::Command::new(python);
        command.env_clear();
        for key in [
            "SystemRoot",
            "WINDIR",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "LOCALAPPDATA",
            "APPDATA",
            "PROGRAMDATA",
            "COMSPEC",
            "NUMBER_OF_PROCESSORS",
        ] {
            if let Some(value) = env::var_os(key) {
                command.env(key, value);
            }
        }
        command
            .args(["-I", "-B", "-u", "-c", WORKER_BOOTSTRAP])
            .arg(&engine)
            .arg(&pipeline)
            .args(["--method", "status", "--params", "{}"])
            .current_dir(&engine)
            .env("GSSTUDIO_RUNTIME", &config)
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
        let mut child = command.spawn().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(45);
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("read-only worker startup timed out");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "worker startup failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let frames: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let response = frames
            .iter()
            .find(|v| v["type"] == "result" && v["id"] == "cli")
            .expect("worker status JSONL result");
        assert!(response["result"]["active"].is_null());
        assert_eq!(response["result"]["cancel_requested"], false);
    }

    #[test]
    fn jsonl_reader_handles_split_utf8_and_eof() {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let mut reader = BufReader::with_capacity(2, "中文\nlast".as_bytes());
                assert_eq!(
                    bounded_line(&mut reader, 20, false)
                        .await
                        .unwrap()
                        .as_deref(),
                    Some("中文\n")
                );
                assert_eq!(
                    bounded_line(&mut reader, 20, false)
                        .await
                        .unwrap()
                        .as_deref(),
                    Some("last")
                );
                assert!(bounded_line(&mut reader, 20, false)
                    .await
                    .unwrap()
                    .is_none());
            });
    }

    #[test]
    fn oversized_protocol_frame_fails_instead_of_growing() {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let mut reader = BufReader::with_capacity(2, &b"123456789\n"[..]);
                assert_eq!(
                    bounded_line(&mut reader, 4, false)
                        .await
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidData
                );
            });
    }

    #[test]
    fn oversized_stderr_is_drained_without_losing_next_line() {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let mut reader = BufReader::with_capacity(2, &b"123456789\nnext\n"[..]);
                let truncated = bounded_line(&mut reader, 5, true).await.unwrap().unwrap();
                assert!(truncated.starts_with("12345"));
                assert!(truncated.contains("truncated"));
                assert_eq!(
                    bounded_line(&mut reader, 5, true).await.unwrap().as_deref(),
                    Some("next\n")
                );
            });
    }
}
