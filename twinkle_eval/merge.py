import glob
import json
import os
from typing import Optional

import numpy as np

from .results_exporters import ResultsExporterFactory

def merge_distributed_results(timestamp: str, hf_repo_id: Optional[str] = None, hf_variant: Optional[str] = "default") -> int:
    """合併平行運算產生的碎片並重新計算評測指標，最後刪除碎片"""

    results_dir = "results"
    json_shards = glob.glob(os.path.join(results_dir, f"results_{timestamp}_node*_rank*.json"))

    if not json_shards:
        print(f"❌ 找不到時間戳記為 {timestamp} 的評測碎片。")
        return 1

    print(f"找到 {len(json_shards)} 個 JSON 配置碎片。開始合併與統整數據...")

    with open(json_shards[0], "r", encoding="utf-8") as f:
        base_result = json.load(f)

    merged_dataset_results = {}
    dataset_file_jsonl_map = {}
    
    for shard_path in json_shards:
        with open(shard_path, "r", encoding="utf-8") as f:
            shard_data = json.load(f)
            
        for ds_name, ds_data in shard_data.get("dataset_results", {}).items():
            if ds_name not in dataset_file_jsonl_map:
                dataset_file_jsonl_map[ds_name] = {}
                
            for file_res in ds_data.get("results", []):
                file_path = file_res["file"]
                if file_path not in dataset_file_jsonl_map[ds_name]:
                    dataset_file_jsonl_map[ds_name][file_path] = {}
                    
                jsonl_paths = file_res.get("individual_runs", {}).get("results", [])
                for run_idx, jsonl_path in enumerate(jsonl_paths):
                    if run_idx not in dataset_file_jsonl_map[ds_name][file_path]:
                        dataset_file_jsonl_map[ds_name][file_path][run_idx] = []
                    if jsonl_path not in dataset_file_jsonl_map[ds_name][file_path][run_idx]:
                        dataset_file_jsonl_map[ds_name][file_path][run_idx].append(jsonl_path)
    
    jsonl_files_merged = set()
    cleared_merged_files = set()
    
    for ds_name, files_map in dataset_file_jsonl_map.items():
        ds_results = []
        for file_path, runs_map in files_map.items():
            run_accuracies = []
            run_merged_jsonl_paths = []
            
            for run_idx, jsonl_paths in runs_map.items():
                all_details = []
                for j_path in jsonl_paths:
                    if os.path.exists(j_path):
                        with open(j_path, "r", encoding="utf-8") as jf:
                            for line in jf:
                                if line.strip():
                                    all_details.append(json.loads(line))
                        jsonl_files_merged.add(j_path)
                
                all_details.sort(key=lambda x: int(x.get("question_id", 0)))
                
                total_correct = sum(1 for d in all_details if d.get("is_correct", False))
                total_questions = len(all_details)
                run_acc = total_correct / total_questions if total_questions > 0 else 0
                run_accuracies.append(run_acc)
                
                merged_jsonl_name = os.path.join(results_dir, f"eval_results_{timestamp}_run{run_idx}.jsonl")
                
                if merged_jsonl_name not in cleared_merged_files:
                    with open(merged_jsonl_name, "w", encoding="utf-8"):
                        pass
                    cleared_merged_files.add(merged_jsonl_name)

                # Append mode since multiple datasets/files may share the same run_idx if single node simulation
                with open(merged_jsonl_name, "a", encoding="utf-8") as jf:
                    for d in all_details:
                        jf.write(json.dumps(d, ensure_ascii=False) + "\n")
                        
                if merged_jsonl_name not in run_merged_jsonl_paths:
                    run_merged_jsonl_paths.append(merged_jsonl_name)
            
            mean_acc = np.mean(run_accuracies) if run_accuracies else 0
            std_acc = np.std(run_accuracies) if len(run_accuracies) > 1 else 0
            
            ds_results.append({
                "file": file_path,
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "individual_runs": {
                    "accuracies": run_accuracies,
                    "results": run_merged_jsonl_paths
                }
            })
            
        ds_avg_acc = np.mean([r["accuracy_mean"] for r in ds_results]) if ds_results else 0
        ds_avg_std = np.mean([r["accuracy_std"] for r in ds_results]) if ds_results else 0
        
        merged_dataset_results[ds_name] = {
            "results": ds_results,
            "average_accuracy": ds_avg_acc,
            "average_std": ds_avg_std
        }
        
    final_results = {
        "timestamp": timestamp,
        "config": base_result["config"],
        "duration_seconds": base_result.get("duration_seconds", 0),
        "dataset_results": merged_dataset_results
    }
    
    base_output_path = os.path.join(results_dir, f"results_{timestamp}")
    exported_files = ResultsExporterFactory.export_results(final_results, base_output_path, ["json", "html"], base_result["config"])
    print(f"✅ 合併完成，結果已匯出至: {', '.join(exported_files)}")
    
    print("🧹 清理 Rank 分散式碎片...")
    for sp in json_shards:
        try:
            os.remove(sp)
        except OSError:
            pass
    for jp in jsonl_files_merged:
        try:
            os.remove(jp)
        except OSError:
            pass
            
    if hf_repo_id:
        try:
            from .hf_uploader import upload_results
            model_name = base_result["config"].get("model", {}).get("name", "unknown_model")
            upload_results(
                repo_id=hf_repo_id,
                variant=hf_variant,
                model_name=model_name,
                results_dir=results_dir,
                timestamp=timestamp,
            )
            print("✅ 成功上傳合併結果至 Hugging Face")
        except Exception as e:
            print(f"❌ 上傳至 Hugging Face 失敗: {e}")
            
    return 0
