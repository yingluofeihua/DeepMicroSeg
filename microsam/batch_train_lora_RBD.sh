#!/bin/bash

# LoRA批量训练脚本 - 修复版
# 用于训练不同train_number参数的SAM LoRA模型

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色和时间戳的信息
print_info() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${RED}[ERROR]${NC} $1"
}

# 基础配置参数
BASE_CONFIG=(
    --train_image_dir "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/train/images"
    --train_mask_dir "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/train/masks"
    --val_image_dir "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/val/images"
    --val_mask_dir "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/val/masks"
    --model_type "vit_b_lm"
    --checkpoint_name "sam_lora_RBD"
    --pretrained_checkpoint "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/checkpoint/fl_best.pt"
    --rank 8
    --attention_layers_to_update 9 10 11
    --gpu_id 3
    --n_epochs 100
    --batch_size 8
    --train_instance_segmentation
    --freeze_prompt_encoder
)

# 基础保存路径
BASE_SAVE_ROOT="/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/RBD"

# 训练数量列表
TRAIN_NUMBERS=(1 2 4 8 16 32 64)

# 确保日志目录存在
LOG_DIR="${BASE_SAVE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/batch_training_$(date +%Y%m%d_%H%M%S).log"

# 记录开始时间
START_TIME=$(date)
print_info "脚本启动，PID: $$"
print_info "批量训练开始时间: ${START_TIME}"
print_info "主日志文件: ${MAIN_LOG}"

# 重定向输出到日志文件（同时保持控制台输出）
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "========================================" 
echo "LoRA批量训练脚本启动"
echo "启动时间: ${START_TIME}"
echo "PID: $$"
echo "========================================"

# 函数：检查环境
check_environment() {
    print_info "检查运行环境..."
    
    # 检查Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        print_info "Python版本: ${PYTHON_VERSION}"
    else
        print_error "Python未找到"
        return 1
    fi
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_info "CUDA可用，检查GPU状态..."
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    else
        print_warning "nvidia-smi未找到，无法检查GPU状态"
    fi
    
    return 0
}

# 函数：检查必要文件和路径
check_prerequisites() {
    print_info "检查必要的文件和路径..."
    
    local error_count=0
    
    # 检查训练脚本
    if [ ! -f "microsam/lora_train.py" ]; then
        print_error "训练脚本不存在: microsam/lora_train.py"
        print_error "当前目录: $(pwd)"
        print_error "目录内容: $(ls -la)"
        error_count=$((error_count + 1))
    else
        print_success "训练脚本存在: microsam/lora_train.py"
    fi
    
    # 检查数据路径
    local data_paths=(
        "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/train/images"
        "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/train/masks"
        "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/val/images"
        "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/val/masks"
    )
    
    for path in "${data_paths[@]}"; do
        if [ ! -d "$path" ]; then
            print_error "数据路径不存在: $path"
            error_count=$((error_count + 1))
        else
            local file_count=$(ls -1 "$path" 2>/dev/null | wc -l)
            print_success "数据路径存在: $path (包含 $file_count 个文件)"
        fi
    done
    
    # 检查预训练检查点
    local checkpoint="/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/checkpoint/fl_best.pt"
    if [ ! -f "$checkpoint" ]; then
        print_error "预训练检查点不存在: $checkpoint"
        error_count=$((error_count + 1))
    else
        print_success "预训练检查点存在: $checkpoint"
    fi
    
    # 检查保存目录权限
    if [ ! -w "$(dirname "$BASE_SAVE_ROOT")" ]; then
        print_error "没有写入权限: $(dirname "$BASE_SAVE_ROOT")"
        error_count=$((error_count + 1))
    else
        mkdir -p "$BASE_SAVE_ROOT"
        print_success "保存目录可访问: $BASE_SAVE_ROOT"
    fi
    
    if [ $error_count -gt 0 ]; then
        print_error "发现 $error_count 个问题，无法继续"
        return 1
    fi
    
    print_success "所有检查通过"
    return 0
}

# 函数：等待GPU内存释放
wait_for_gpu_memory() {
    local max_wait=300  # 最大等待时间（秒）
    local wait_time=0
    
    print_info "等待GPU内存释放..."
    
    while [ $wait_time -lt $max_wait ]; do
        if command -v nvidia-smi &> /dev/null; then
            local gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 2>/dev/null)
            if [ $? -eq 0 ] && [ "$gpu_memory" -lt 1000 ]; then  # 小于1GB认为可用
                print_success "GPU内存已释放 (${gpu_memory}MB)，可以开始下一个训练"
                return 0
            fi
            print_info "GPU内存使用: ${gpu_memory}MB，等待释放... (${wait_time}/${max_wait}秒)"
        fi
        
        sleep 10
        wait_time=$((wait_time + 10))
    done
    
    print_warning "等待GPU内存释放超时，继续执行"
    return 1
}

# 函数：单个训练任务
run_single_training() {
    local train_number=$1
    local save_root="${BASE_SAVE_ROOT}/b_lm_sam_lora_fl_number_${train_number}"
    local log_file="${LOG_DIR}/training_number_${train_number}_$(date +%Y%m%d_%H%M%S).log"
    
    print_info "=========================================="
    print_info "开始训练任务: train_number=${train_number}"
    print_info "保存路径: ${save_root}"
    print_info "任务日志: ${log_file}"
    
    # 创建保存目录
    mkdir -p "${save_root}"
    
    # 构建完整命令
    local cmd="python microsam/lora_train.py"
    for arg in "${BASE_CONFIG[@]}"; do
        cmd="${cmd} ${arg}"
    done
    cmd="${cmd} --save_root ${save_root} --train_number ${train_number}"
    
    print_info "执行命令: ${cmd}"
    
    # 记录开始时间
    local task_start_time=$(date)
    print_info "任务开始时间: ${task_start_time}"
    
    # 执行训练（同步执行，等待完成）
    echo "开始训练 train_number=${train_number} 于 ${task_start_time}" > "${log_file}"
    echo "命令: ${cmd}" >> "${log_file}"
    echo "==========================================" >> "${log_file}"
    
    # 直接执行命令并等待完成
    ${cmd} >> "${log_file}" 2>&1
    local exit_code=$?
    
    local task_end_time=$(date)
    print_info "任务结束时间: ${task_end_time}"
    
    # 记录结果
    echo "==========================================" >> "${log_file}"
    echo "训练结束于 ${task_end_time}" >> "${log_file}"
    echo "退出码: ${exit_code}" >> "${log_file}"
    
    if [ $exit_code -eq 0 ]; then
        print_success "train_number=${train_number} 训练完成"
        return 0
    else
        print_error "train_number=${train_number} 训练失败，退出码: ${exit_code}"
        print_error "请查看日志: ${log_file}"
        return 1
    fi
}

# 函数：清理GPU缓存
cleanup_gpu() {
    print_info "清理GPU缓存..."
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU缓存已清理')
else:
    print('CUDA不可用')
" 2>/dev/null || print_warning "GPU清理失败"
}

# 主训练循环
main() {
    local success_count=0
    local total_count=${#TRAIN_NUMBERS[@]}
    
    print_info "开始批量训练，总共 ${total_count} 个任务"
    print_info "训练数量列表: ${TRAIN_NUMBERS[*]}"
    
    # 环境检查
    if ! check_environment; then
        print_error "环境检查失败"
        return 1
    fi
    
    if ! check_prerequisites; then
        print_error "先决条件检查失败"
        return 1
    fi
    
    # 开始训练循环
    for i in "${!TRAIN_NUMBERS[@]}"; do
        local train_number=${TRAIN_NUMBERS[$i]}
        local current_task=$((i + 1))
        
        print_info "=========================================="
        print_info "训练任务 ${current_task}/${total_count}: train_number=${train_number}"
        
        # 等待GPU内存释放（除了第一个任务）
        if [ $i -gt 0 ]; then
            wait_for_gpu_memory
            cleanup_gpu
            print_info "任务间等待30秒..."
            sleep 30
        fi
        
        # 执行训练
        if run_single_training "$train_number"; then
            success_count=$((success_count + 1))
            print_success "任务 ${train_number} 成功完成 (${success_count}/${total_count})"
        else
            print_error "任务 ${train_number} 失败"
            # 在后台运行时不询问，直接继续
            print_warning "继续下一个训练任务..."
        fi
        
        # 显示进度
        local progress=$((current_task * 100 / total_count))
        print_info "总体进度: ${progress}% (${current_task}/${total_count})"
    done
    
    # 训练总结
    local end_time=$(date)
    print_info "=========================================="
    print_success "批量训练完成!"
    print_info "开始时间: ${START_TIME}"
    print_info "结束时间: ${end_time}"
    print_info "成功完成: ${success_count}/${total_count} 个任务"
    
    # 显示失败的任务
    if [ $success_count -lt $total_count ]; then
        print_warning "以下任务可能需要检查:"
        for train_number in "${TRAIN_NUMBERS[@]}"; do
            local save_root="${BASE_SAVE_ROOT}/b_lm_sam_lora_fl_number_${train_number}"
            if [ ! -d "${save_root}" ] || [ -z "$(ls -A ${save_root} 2>/dev/null)" ]; then
                print_warning "  - train_number=${train_number}"
            fi
        done
    fi
    
    print_info "主日志: ${MAIN_LOG}"
    print_info "各任务日志在: ${LOG_DIR}/"
}

# 脚本入口点 - 移除交互式确认
print_info "LoRA批量训练脚本启动"
print_info "配置信息:"
print_info "  - 训练数量: ${TRAIN_NUMBERS[*]}"
print_info "  - GPU ID: 1"
print_info "  - 总轮数: 100"
print_info "  - 批大小: 8"
print_info "  - 保存根目录: ${BASE_SAVE_ROOT}"

# 直接开始执行
main

print_info "脚本执行完毕"