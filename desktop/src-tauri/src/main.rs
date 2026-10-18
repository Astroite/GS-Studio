#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(windows)]
mod native_job;
mod supervisor;

use serde_json::Value;
use supervisor::EngineSupervisor;
use tauri::Manager;

#[tauri::command]
async fn engine_call(
    method: String,
    params: Value,
    app: tauri::AppHandle,
    state: tauri::State<'_, EngineSupervisor>,
) -> Result<Value, String> {
    state.call(&app, method, params).await
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(EngineSupervisor::default())
        .invoke_handler(tauri::generate_handler![engine_call])
        .build(tauri::generate_context!())
        .expect("Unable to initialize GS Studio desktop");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            tauri::async_runtime::block_on(app.state::<EngineSupervisor>().shutdown());
        }
    });
}
