import json
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_null_mask_files(json_files, output_json_path):
    """
    查找所有 info['mask_file'] 為 None 的數據項，並將它們的信息保存到 JSON 文件中
    
    Args:
        json_files: JSON 映射文件列表
        output_json_path: 輸出 JSON 文件的路徑
    """
    null_mask_items = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset_name = data.get('dataset_name', Path(json_file).stem)
            images_path = data['images_path']
            
            for img_name, info in data['mapping'].items():
                if 'mask_file' not in info or info['mask_file'] is None:
                    null_mask_items.append({
                        'dataset_name': dataset_name,
                        'json_file': json_file,
                        'image_name': img_name,
                        'image_path': str(Path(images_path) / img_name),
                        'info': info
                    })
                    logging.info(f"找到 mask_file 為 None 的項目: {dataset_name}/{img_name}")
        
        except Exception as e:
            logging.error(f"處理 {json_file} 時出錯: {e}")
    
    # 將結果保存到 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(null_mask_items, f, indent=2)
    
    logging.info(f"找到 {len(null_mask_items)} 個 mask_file 為 None 的項目")
    logging.info(f"結果已保存到 {output_json_path}")
    
    return null_mask_items

if __name__ == "__main__":
    # JSON 映射文件列表，與您的 example_usage 函數中相同
    json_files = [
        "/LD-FS/data/public_dataset/Retrain/mappings/YIM_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Omnipose_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Hoechst_33342-stained_nuclei_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Breast_cancer_cell_dataset_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Fluorescent_Neuronal_Cells_1.0_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/MDA-MB-231&BT20_cells_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Stardist_AsPC1_from_TC_mapping.json"
    ]
    
    # 輸出 JSON 文件路徑
    output_json_path = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/public_dataset/convertedtif/null_mask_files.json"
    
    # 查找並保存 mask_file 為 None 的項目
    null_mask_items = find_null_mask_files(json_files, output_json_path)
    
    # 打印摘要信息
    print(f"總共找到 {len(null_mask_items)} 個 mask_file 為 None 的項目")
    print(f"詳細信息已保存到 {output_json_path}")