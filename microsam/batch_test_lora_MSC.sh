#!/bin/bash

# LoRA批量测试脚本
# 用于测试不同train_number参数训练的SAM LoRA模型

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

# 基础配置参数（固定部分）
BASE_TEST_CONFIG=(
    --test_image_dir "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/test/images"
    --test_mask_dir "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/test/masks"
    --model_type "vit_b_lm"
    --gpu_id 1
    --rank 8
    --attention_layers_to_update 9 10 11
)

MODEL_TYPE='base'

# 基础路径
BASE_RESULTS_ROOT="/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/MSC"

# 测试数量列表
TEST_NUMBERS=(1 2 4 8 16 32 64)

# 日志配置
LOG_DIR="${BASE_RESULTS_ROOT}/test_logs"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/batch_testing_$(date +%Y%m%d_%H%M%S).log"

# 记录开始时间
START_TIME=$(date)
print_info "批量测试开始时间: ${START_TIME}"
print_info "主日志文件: ${MAIN_LOG}"

# 重定向输出到日志文件
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "========================================" 
echo "LoRA批量测试脚本启动"
echo "启动时间: ${START_TIME}"
echo "PID: $$"
echo "========================================"

# 函数：检查环境
check_environment() {
    print_info "检查测试环境..."
    
    # 检查Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        print_info "Python版本: ${PYTHON_VERSION}"
    else
        print_error "Python未找到"
        return 1
    fi
    
    # 检查CUDA和指定GPU
    if command -v nvidia-smi &> /dev/null; then
        print_info "检查GPU 4状态..."
        GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader --id=4 2>/dev/null)
        if [ $? -eq 0 ]; then
            print_info "GPU 4 信息: ${GPU_INFO}"
        else
            print_warning "GPU 4 不可用或未找到"
        fi
    else
        print_warning "nvidia-smi未找到，无法检查GPU状态"
    fi
    
    return 0
}

# 函数：检查测试脚本和数据路径
check_prerequisites() {
    print_info "检查必要的文件和路径..."
    
    local error_count=0
    
    # 检查测试脚本
    if [ ! -f "microsam/lora_test.py" ]; then
        print_error "测试脚本不存在: microsam/lora_test.py"
        print_error "当前目录: $(pwd)"
        error_count=$((error_count + 1))
    else
        print_success "测试脚本存在: microsam/lora_test.py"
    fi
    
    # 检查测试数据路径
    local test_paths=(
        "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/test/images"
        "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/test/masks"
    )
    
    for path in "${test_paths[@]}"; do
        if [ ! -d "$path" ]; then
            print_error "测试数据路径不存在: $path"
            error_count=$((error_count + 1))
        else
            local file_count=$(ls -1 "$path" 2>/dev/null | wc -l)
            print_success "测试数据路径存在: $path (包含 $file_count 个文件)"
        fi
    done
    
    if [ $error_count -gt 0 ]; then
        print_error "发现 $error_count 个问题，无法继续"
        return 1
    fi
    
    print_success "先决条件检查通过"
    return 0
}

# 函数：检查模型检查点是否存在
check_checkpoint() {
    local train_number=$1
    local checkpoint_path="${BASE_RESULTS_ROOT}/b_lm_sam_lora_${MODEL_TYPE}_number_${train_number}/checkpoints/sam_lora_MSC/best.pt"
    
    if [ ! -f "$checkpoint_path" ]; then
        print_error "检查点不存在: $checkpoint_path"
        return 1
    else
        local checkpoint_size=$(ls -lh "$checkpoint_path" | awk '{print $5}')
        print_success "检查点存在: $checkpoint_path (大小: $checkpoint_size)"
        return 0
    fi
}

# 函数：单个测试任务
run_single_test() {
    local train_number=$1
    local checkpoint_path="${BASE_RESULTS_ROOT}/b_lm_sam_lora_${MODEL_TYPE}_number_${train_number}/checkpoints/sam_lora_MSC/best.pt"
    local results_dir="${BASE_RESULTS_ROOT}/b_lm_sam_lora_${MODEL_TYPE}_number_${train_number}/test"
    local log_file="${LOG_DIR}/test_MSC_number_${train_number}_$(date +%Y%m%d_%H%M%S).log"
    
    print_info "=========================================="
    print_info "开始测试任务: train_number=${train_number}"
    print_info "检查点路径: ${checkpoint_path}"
    print_info "结果保存路径: ${results_dir}"
    print_info "测试日志: ${log_file}"
    
    # 检查检查点是否存在
    if ! check_checkpoint "$train_number"; then
        print_error "跳过 train_number=${train_number}，检查点不存在"
        return 1
    fi
    
    # 创建结果目录
    mkdir -p "${results_dir}"
    
    # 构建完整命令
    local cmd="python microsam/lora_test.py"
    cmd="${cmd} --best_checkpoint ${checkpoint_path}"
    cmd="${cmd} --results_dir ${results_dir}"
    
    # 添加其他固定参数
    for arg in "${BASE_TEST_CONFIG[@]}"; do
        cmd="${cmd} ${arg}"
    done
    
    print_info "执行命令: ${cmd}"
    
    # 记录开始时间
    local task_start_time=$(date)
    print_info "测试开始时间: ${task_start_time}"
    
    # 执行测试
    echo "开始测试 train_number=${train_number} 于 ${task_start_time}" > "${log_file}"
    echo "检查点: ${checkpoint_path}" >> "${log_file}"
    echo "命令: ${cmd}" >> "${log_file}"
    echo "==========================================" >> "${log_file}"
    
    # 执行命令并等待完成
    ${cmd} >> "${log_file}" 2>&1
    local exit_code=$?
    
    local task_end_time=$(date)
    print_info "测试结束时间: ${task_end_time}"
    
    # 记录结果
    echo "==========================================" >> "${log_file}"
    echo "测试结束于 ${task_end_time}" >> "${log_file}"
    echo "退出码: ${exit_code}" >> "${log_file}"
    
    if [ $exit_code -eq 0 ]; then
        print_success "train_number=${train_number} 测试完成"
        
        # 检查结果文件
        if [ -d "${results_dir}" ] && [ "$(ls -A ${results_dir} 2>/dev/null)" ]; then
            local result_count=$(ls -1 "${results_dir}" 2>/dev/null | wc -l)
            print_success "生成了 ${result_count} 个结果文件"
        fi
        
        return 0
    else
        print_error "train_number=${train_number} 测试失败，退出码: ${exit_code}"
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

# 函数：生成测试摘要
generate_summary() {
    local success_count=$1
    local total_count=$2
    
    print_info "生成测试摘要..."
    
    local summary_file="${LOG_DIR}/test_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "LoRA批量测试摘要" > "${summary_file}"
    echo "===================" >> "${summary_file}"
    echo "测试时间: ${START_TIME} - $(date)" >> "${summary_file}"
    echo "成功: ${success_count}/${total_count}" >> "${summary_file}"
    echo "" >> "${summary_file}"
    
    for train_number in "${TEST_NUMBERS[@]}"; do
        local results_dir="${BASE_RESULTS_ROOT}/b_lm_sam_lora_${MODEL_TYPE}_number_${train_number}/test"
        if [ -d "${results_dir}" ] && [ "$(ls -A ${results_dir} 2>/dev/null)" ]; then
            local result_count=$(ls -1 "${results_dir}" 2>/dev/null | wc -l)
            echo "✅ train_number=${train_number}: ${result_count} 个结果文件" >> "${summary_file}"
        else
            echo "❌ train_number=${train_number}: 测试失败或无结果" >> "${summary_file}"
        fi
    done
    
    print_success "测试摘要保存到: ${summary_file}"
    cat "${summary_file}"
}

# 主测试循环
main() {
    local success_count=0
    local total_count=${#TEST_NUMBERS[@]}
    
    print_info "开始批量测试，总共 ${total_count} 个任务"
    print_info "测试数量列表: ${TEST_NUMBERS[*]}"
    
    # 环境检查
    if ! check_environment; then
        print_error "环境检查失败"
        return 1
    fi
    
    if ! check_prerequisites; then
        print_error "先决条件检查失败"
        return 1
    fi
    
    # 开始测试循环
    for i in "${!TEST_NUMBERS[@]}"; do
        local train_number=${TEST_NUMBERS[$i]}
        local current_task=$((i + 1))
        
        print_info "=========================================="
        print_info "测试任务 ${current_task}/${total_count}: train_number=${train_number}"
        
        # 清理GPU缓存（在每个任务开始前）
        if [ $i -gt 0 ]; then
            cleanup_gpu
            print_info "任务间等待10秒..."
            sleep 10
        fi
        
        # 执行测试
        if run_single_test "$train_number"; then
            success_count=$((success_count + 1))
            print_success "任务 ${train_number} 成功完成 (${success_count}/${total_count})"
        else
            print_error "任务 ${train_number} 失败"
        fi
        
        # 显示进度
        local progress=$((current_task * 100 / total_count))
        print_info "总体进度: ${progress}% (${current_task}/${total_count})"
    done
    
    # 测试总结
    local end_time=$(date)
    print_info "=========================================="
    print_success "批量测试完成!"
    print_info "开始时间: ${START_TIME}"
    print_info "结束时间: ${end_time}"
    print_info "成功完成: ${success_count}/${total_count} 个任务"
    
    # 生成详细摘要
    generate_summary "$success_count" "$total_count"
    
    print_info "主日志: ${MAIN_LOG}"
    print_info "各任务日志在: ${LOG_DIR}/"
    
    # 显示结果目录
    print_info "测试结果保存在以下目录:"
    for train_number in "${TEST_NUMBERS[@]}"; do
        local results_dir="${BASE_RESULTS_ROOT}/b_lm_sam_lora_${MODEL_TYPE}_number_${train_number}/test"
        if [ -d "${results_dir}" ]; then
            print_info "  - ${results_dir}"
        fi
    done
}

# 脚本入口点
print_info "LoRA批量测试脚本启动"
print_info "配置信息:"
print_info "  - 测试数量: ${TEST_NUMBERS[*]}"
print_info "  - GPU ID: 4"
print_info "  - 模型类型: vit_b_lm"
print_info "  - 基础结果路径: ${BASE_RESULTS_ROOT}"

# 直接开始执行
main

print_info "脚本执行完毕"