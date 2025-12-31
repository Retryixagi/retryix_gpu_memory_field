# RetryIX-Py: Open-Source High-Performance Computing Backend

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/retryix-py.svg)](https://pypi.org/project/retryix-py/)

## Release: Memory‑Field Core v0.2.0

**2026年01月01日** - 本次發佈為 Memory‑Field Core 的穩定釋出（記憶場驅動最小核心與驗證流水線）。本版本聚焦於「記憶場」(memory‑driven) 的最小運算核心、SPIR‑V shader 工具、以及自動化上傳與驗證流程；早期的雙矩陣（dual‑matrix）研發工作為實驗性內容，已移入備份分支並非本次釋出的一部分。

### 本次釋出重點

- **記憶場核心（memory‑driven minimal kernel）**：小型、可重複驗證的 field compute kernel
- **Shader 編譯與工具鏈**：包含 `tools/compile_minimal_shader.py` 與 `build_shaders.bat` 的編譯輔助
- **自動化上傳/驗證流水線**：GitHub Actions + S3 上傳腳本（`tools/s3_upload_artifacts.py`）
- **驗證資料與報告**：`artifacts/mf_r4_repeat10.csv`、`artifacts/mf_r4_repeat30.csv` 與 `report/memory_field_summary.pdf`

> 注意：本次釋出僅包含 GPU‑only 的記憶場核心（single‑matrix）。其他未公開或實驗性研究內容已移至內部管理，未包含於公開釋出。

---

RetryIX-Py provides Python bindings for the RetryIX computing framework, enabling hardware-agnostic GPU computing with automatic CPU fallback.

## 🚀 Key Features

- **場對齊異構處理**: CPU/GPU通過場拓樸對齊實現協作處理
- **94%+對齊置信度**: 異構矩陣表達同一語義場的精確度
- **真實DRAM記憶體池**: 系統級記憶體管理，完全真實實現
- **多模態學習處理**: 支援圖片、文字等多模態內容處理
- **整合應用生態**: Python應用、C語言服務、混合學習系統完整工具套件
- **Hardware Agnostic**: Unified API supporting NVIDIA CUDA, AMD ROCM, Intel Level Zero
- **Automatic Fallback**: Graceful CPU fallback when GPU is unavailable
- **SVM Support**: Shared Virtual Memory for zero-copy data transfer
- **Tile-Based Operations**: Optimized matrix operations for modern GPUs
- **Semantic Computing**: Advanced algorithms for AI and scientific computing
- **Cross-Platform**: Windows, Linux, and macOS support

## 📁 專案結構

```
F:\1213\                           # 專案根目錄
├── retryix_workspace\            # 🏗️ RetryIX 專用工作區
│   ├── tools\                    # 🛠️ 核心工具組件
│   ├── pipelines\                # 🔄 工作流程與演示
│   ├── toolchain\                # ⚙️ 工具鏈管理系統
│   ├── c_ai_backend\             # 💻 C語言AI後端
│   └── examples\                 # 📚 範例程式
├── retryix_launcher.py           # 🚀 快速啟動腳本
├── PROJECT_STRUCTURE.md          # 📖 詳細結構說明
├── src\                          # 📁 原始源代碼
├── backend\                      # 🔧 C後端編譯檔案
├── test\                         # 🧪 測試檔案
└── [其他項目文件...]            # 其他檔案
```
│   ├── retryix_dashboard.py           # 網頁儀表板
│   ├── retryix_launcher_v2.bat        # 進階啟動器
│   ├── integrated_pipeline.bat        # 整合流水線
│   ├── README_TOOLCHAIN.md           # 工具鏈說明
│   └── RETRYIX_TOOLCHAIN_COMPLETION_REPORT.md
├── c_ai_backend/                 # C語言AI後端
├── backend/                      # 核心後端實現
├── retryix/                      # Python包
├── test/                         # 測試文件
├── include/                      # 頭文件
├── src/                          # 源碼
└── retryix_launcher.bat          # 🚀 快速啟動器 (根目錄)
```

### 🎮 快速開始

#### 啟動統一工具鏈
```batch
# 從根目錄啟動
retryix_launcher.bat
```

#### 啟動網頁儀表板
```batch
python toolchain\retryix_dashboard.py
```

#### 運行完整測試
```batch
python toolchain\retryix_toolchain_manager.py workflow --workflow full_test
```
```

### From Source

```bash
# Clone the repository
git clone https://github.com/retryix/retryix-py.git
cd retryix-py

# Install dependencies
pip install -r requirements.txt

# Build and install
pip install -e .
```

### System Requirements

- Python 3.8+
- C/C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- Optional: GPU drivers for hardware acceleration
  - NVIDIA: CUDA Toolkit
  - AMD: ROCM
  - Intel: Level Zero

## � Quick Start

### 使用啟動腳本 (推薦)

```bash
# 運行學習演示
python retryix_launcher.py demo

# 運行完整工作流程
python retryix_launcher.py workflow --workflow learning_demo
```

### 直接Python匯入

```python
# 匯入工具鏈管理器
from retryix_workspace.toolchain.retryix_toolchain_manager import RetryIXToolchainManager

# 創建管理器實例
manager = RetryIXToolchainManager()

# 運行學習演示工作流程
result = manager.run_integrated_workflow('learning_demo')
```

### 單獨使用組件

```python
# 匯入場對齊系統
from retryix_workspace.tools.field_alignment_system import FieldAlignmentSystem

# 創建實例並運行
fas = FieldAlignmentSystem()
fas.run_analysis()
```
import retryix_py as rx
import numpy as np

# Create matrices
a = np.random.randn(1000, 1000).astype(np.float32)
b = np.random.randn(1000, 1000).astype(np.float32)
c = np.zeros((1000, 1000), dtype=np.float32)

# Perform high-performance matrix multiplication
rx.matmul(a, b, c)

print("Matrix multiplication completed!")
print(f"Result shape: {c.shape}")
```

## 🔌 GitHub App — 安裝與設定 (Setup URL)

若你要使用本專案提供的 GitHub App（Retryix GPU Memory Field），請依下列步驟完成安裝與後設置：

1. 前往 GitHub App 頁面並安裝：
   - App URL: https://github.com/apps/retryix-gpu-memory-field
   - 點選 **Install** → 選擇要安裝到的帳號或組織 → 選擇要授權的 repository（可先選單一測試 repository）。

2. 安裝後請設定 **Setup URL**（會導向此 README）：
   - 建議填入本 repo README（或專門的 setup 頁面）。
   - 此連結會在使用者安裝 App 後顯示，方便導引完成後續設定。

3. 在 repository 的 Settings → Secrets 中新增下列 Secrets（或使用 OIDC + AssumeRole）：
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`（例如 `ap-southeast-2`）
   - `S3_BUCKET`（例如 `memory-field-engine-baselines`）
   - `S3_PREFIX`（可選，用來分隔上傳路徑）

4. 驗證與測試：
   - 使用 `workflow_dispatch` 手動觸發 `.github/workflows/upload-artifacts.yml`，確認 artifact 能正確上傳到 S3。
   - 若需即時事件觸發，可設定 webhook（需 HTTPS endpoint 與 secret）或讓 Engine 發出 `repository_dispatch` 事件。

> 備註：推薦使用 **OIDC + IAM Role** 以避免在 Secrets 中儲存長期 AWS 金鑰（更安全）。

## 🎯 Advanced Usage

### Semantic Field Repair

```python
from retryix_semantic_repair import RetryIX_GPUSemanticRepair
import numpy as np

# Initialize the repair system
repair = RetryIX_GPUSemanticRepair(
    total_nodes=2048,
    node_attributes=16,
    semantic_layers=8
)

# Create noisy semantic field
noisy_field = np.random.randn(2048, 16).astype(np.float32)

# Repair the semantic field
result = repair.run_semantic_field_repair(
    input_data=noisy_field,
    max_cycles=10,
    convergence_threshold=0.01
)

print(f"Repair completed in {result['cycles_completed']} cycles")
print(f"Final coherence: {result['final_coherence']:.4f}")
```

### Hardware Detection

```python
import retryix_py as rx

# Check available hardware
print("Available backends:")
backends = rx.get_available_backends()
for backend in backends:
    print(f"- {backend}")

# Get current backend
current = rx.get_current_backend()
print(f"Current backend: {current}")
```

### 場對齊異構處理

```python
from field_alignment_system import FieldAlignmentSystem, FieldType, MatrixWorkMode
import numpy as np

# 初始化場對齊系統
field_system = FieldAlignmentSystem()

# 創建測試場拓樸
matrix = np.random.rand(64, 64)
topology = field_system.extract_field_topology(
    matrix, MatrixWorkMode.CPU_FIXED, FieldType.SEMANTIC
)

print(f"場類型: {topology.field_type.value}")
print(f"關鍵點數量: {len(topology.critical_points)}")

# 創建異構矩陣對（CPU固定 + GPU動態）
cpu_expr, gpu_expr = field_system.create_heterogeneous_field_pair(
    topology, cpu_size=32, gpu_size=128
)

print(f"CPU矩陣形狀: {cpu_expr.matrix.shape}")
print(f"GPU矩陣形狀: {gpu_expr.matrix.shape}")

# 驗證場對齊
cpu_topology = field_system.extract_field_topology(
    cpu_expr.matrix, cpu_expr.work_mode, topology.field_type
)
gpu_topology = field_system.extract_field_topology(
    gpu_expr.matrix, gpu_expr.work_mode, topology.field_type
)

alignment = field_system.align_fields(cpu_topology, gpu_topology)
print(f"場對齊成功: {alignment['aligned']}")
print(f"對齊置信度: {alignment['confidence']:.2%}")
```

## 🏗️ Architecture

RetryIX uses a modular architecture with multiple bridge implementations:

- **CUDA Bridge**: For NVIDIA GPUs
- **ROCM Bridge**: For AMD GPUs
- **Intel L0 Bridge**: For Intel GPUs
- **CPU Bridge**: Fallback for all systems

The system automatically detects and selects the optimal backend at runtime.

## 📚 Documentation

- [API Reference](docs/api.md)
- [Hardware Setup](docs/hardware.md)
- [Performance Tuning](docs/performance.md)
- [Contributing](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run unit tests
python -m pytest

# Run performance benchmarks
python test_matmul.py

# Test semantic repair
python test_tensorflow_gpu_semantic_repair.py

# Test 場對齊異構處理系統
python test_field_aligned_heterogeneous_system.py

# Test 場對齊流水線
cd test/pipeline && python start_field_aligned_pipeline.py

# Test 多模態處理
python test_image_processing_with_token_cost.py

# 🚀 運行整合流水線應用
cd test/pipeline && python run_pipeline_apps.py list    # 列出所有Python應用
cd test/pipeline && python run_pipeline_apps.py run image_processing    # 運行圖片處理測試
cd test/pipeline && python run_pipeline_apps.py run hybrid_learning     # 運行混合學習系統
cd test/pipeline && python run_pipeline_apps.py run student_demo        # 運行學生模型演示
cd test/pipeline && python run_pipeline_apps.py run performance_comparison  # 運行性能比較
cd test/pipeline && python run_pipeline_apps.py run matrix_4096_test    # 運行4096矩陣測試
cd test/pipeline && python run_pipeline_apps.py run-all                 # 運行所有Python應用

# 🔧 管理C語言應用服務
cd test/pipeline && python manage_c_services.py list     # 列出所有C應用服務
cd test/pipeline && python manage_c_services.py check-all # 檢查所有C應用檔案
cd test/pipeline && python manage_c_services.py build --app dram_service  # 編譯DRAM服務
cd test/pipeline && python manage_c_services.py run --app dram_service    # 運行DRAM服務
cd test/pipeline && python manage_c_services.py build-all  # 編譯所有可編譯應用
```
python test_retryix_semantic_repair.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the principles of open-source computing
- Inspired by the need for hardware-agnostic high-performance computing
- Thanks to all contributors and the open-source community

## 📞 Support

- Issues: [GitHub Issues](https://github.com/retryix/retryix-py/issues)
- Discussions: [GitHub Discussions](https://github.com/retryix/retryix-py/discussions)
- Email: contact@retryix.org

---

**RetryIX**: Democratizing high-performance computing through open-source innovation.

## License
This project is licensed under the Retryix Limited Research License (RL-1.0).
For commercial/institutional licensing inquiries contact: ixu@retryixagi.com
