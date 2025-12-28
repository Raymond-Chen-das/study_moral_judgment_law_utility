"""
環境設定檢查腳本
檢查所有必要的套件、資料夾結構與資料檔案是否就位
"""

import sys
import importlib
from pathlib import Path
from typing import Tuple, List

def print_section(title: str):
    """印出區段標題"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_subsection(title: str):
    """印出子區段標題"""
    print(f"\n{title}")
    print("-" * 70)

def check_python_version() -> bool:
    """
    檢查Python版本
    
    Returns:
        bool: 版本是否符合要求（>= 3.8）
    """
    print_subsection("1. Python 版本檢查")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python 版本過低: {version_str}")
        print(f"     最低需求: Python 3.8")
        print(f"     建議版本: Python 3.10+")
        return False
    
    print(f"  ✓ Python {version_str}")
    
    if version.minor >= 10:
        print(f"     (推薦版本，相容性良好)")
    elif version.minor >= 8:
        print(f"     (符合最低需求)")
    
    return True

def check_packages() -> Tuple[bool, List[str]]:
    """
    檢查必要套件是否已安裝
    
    Returns:
        Tuple[bool, List[str]]: (是否全部安裝, 缺少的套件列表)
    """
    print_subsection("2. Python 套件檢查")
    
    # 定義必要套件（模組名稱 -> 套件名稱）
    required_packages = {
        # 資料處理
        'pandas': ('pandas', '2.0.0'),
        'numpy': ('numpy', '1.24.0'),
        
        # 視覺化
        'plotly': ('plotly', '5.18.0'),
        'seaborn': ('seaborn', '0.12.0'),
        'matplotlib': ('matplotlib', '3.7.0'),
        
        # 統計分析
        'scipy': ('scipy', '1.11.0'),
        'statsmodels': ('statsmodels', '0.14.0'),
        
        # 機器學習
        'sklearn': ('scikit-learn', '1.3.0'),
        'xgboost': ('xgboost', '2.0.0'),
        
        # 可解釋性
        'shap': ('shap', '0.44.0'),
        
        # 其他工具
        'yaml': ('pyyaml', '6.0'),
        'openpyxl': ('openpyxl', '3.1.0'),
        'tqdm': ('tqdm', '4.66.0'),
    }
    
    missing = []
    installed = []
    version_issues = []
    
    for module_name, (package_name, min_version) in required_packages.items():
        try:
            mod = importlib.import_module(module_name)
            
            # 嘗試取得版本號
            version = 'unknown'
            if hasattr(mod, '__version__'):
                version = mod.__version__
            elif hasattr(mod, 'VERSION'):
                version = mod.VERSION
            
            print(f"  ✓ {package_name:20s} (版本: {version})")
            installed.append(package_name)
            
        except ImportError:
            print(f"  ❌ {package_name:20s} (未安裝)")
            missing.append(package_name)
    
    # 總結
    print(f"\n  已安裝: {len(installed)}/{len(required_packages)}")
    
    if missing:
        print(f"\n  ⚠️  缺少 {len(missing)} 個套件")
        print(f"\n  請執行以下指令安裝:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\n  或一次安裝所有套件:")
        print(f"  pip install -r requirements.txt")
        return False, missing
    
    return True, []

def check_directory_structure() -> bool:
    """
    檢查資料夾結構是否完整
    
    Returns:
        bool: 結構是否完整
    """
    print_subsection("3. 資料夾結構檢查")
    
    # 定義必要的資料夾
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/metadata',
        'src/data',
        'src/analysis/descriptive',
        'src/analysis/clustering',
        'src/analysis/inference',
        'src/modeling',
        'src/visualization',
        'src/utils',
        'scripts',
        'outputs/figures/chapter3_exploration',
        'outputs/figures/chapter4_inference',
        'outputs/figures/chapter5_modeling',
        'outputs/tables/chapter3',
        'outputs/tables/chapter4',
        'outputs/tables/chapter5',
        'outputs/models',
        'outputs/logs',
        'report/figures',
        'report/tables',
        'report/drafts',
        'references/papers',
        'references/philosophy',
        'references/statistical_methods',
        'config',
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n  ⚠️  缺少 {len(missing_dirs)} 個資料夾")
        print(f"\n  請執行 PowerShell 腳本重新建立:")
        print(f"  .\\setup_project_structure.ps1")
        return False
    
    print(f"\n  ✓ 所有資料夾結構完整")
    return True

def check_data_files() -> Tuple[bool, List[str]]:
    """
    檢查必要的資料檔案是否存在
    
    Returns:
        Tuple[bool, List[str]]: (是否全部存在, 缺少的檔案列表)
    """
    print_subsection("4. 資料檔案檢查")
    
    data_dir = Path('data/raw')
    
    # 必要的資料檔案
    required_files = {
        'SharedResponsesSurvey.csv': {
            'description': '主要問卷資料（含人口統計變數）',
            'expected_size_mb': (200, 350),  # 預期大小範圍
        },
        'CountriesChangePr.csv': {
            'description': '國家層級道德偏好AMCE值',
            'expected_size_mb': (0.01, 0.05),
        },
        'country_cluster_map.csv': {
            'description': '國家文化圈分類',
            'expected_size_mb': (0.001, 0.01),
        },
        'moral_distance.csv': {
            'description': '國家間道德距離矩陣',
            'expected_size_mb': (0.001, 0.01),
        },
    }
    
    # 可選的資料檔案
    optional_files = {
        'dendrogram_Culture.csv': '文化樹狀圖資料',
        'MMdata_ReadMe.txt': '資料集說明文件',
    }
    
    missing_files = []
    found_files = []
    size_warnings = []
    
    print("  必要檔案:")
    for filename, info in required_files.items():
        filepath = data_dir / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            min_size, max_size = info['expected_size_mb']
            
            if min_size <= size_mb <= max_size:
                print(f"    ✓ {filename:30s} ({size_mb:.1f} MB)")
                print(f"      └─ {info['description']}")
                found_files.append(filename)
            else:
                print(f"    ⚠️  {filename:30s} ({size_mb:.1f} MB)")
                print(f"      └─ 檔案大小異常（預期: {min_size}-{max_size} MB）")
                size_warnings.append(filename)
                found_files.append(filename)
        else:
            print(f"    ❌ {filename:30s} (缺少)")
            print(f"      └─ {info['description']}")
            missing_files.append(filename)
    
    print("\n  可選檔案:")
    for filename, description in optional_files.items():
        filepath = data_dir / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"    ✓ {filename:30s} ({size_mb:.1f} MB)")
            print(f"      └─ {description}")
        else:
            print(f"    - {filename:30s} (無)")
            print(f"      └─ {description}")
    
    # 總結
    print(f"\n  必要檔案: {len(found_files)}/{len(required_files)}")
    
    if missing_files:
        print(f"\n  ⚠️  缺少 {len(missing_files)} 個必要檔案")
        print(f"\n  請至以下網址下載:")
        print(f"  https://osf.io/3hvt2/")
        print(f"\n  下載後請放置於: {data_dir}/")
        return False, missing_files
    
    if size_warnings:
        print(f"\n  ⚠️  {len(size_warnings)} 個檔案大小異常，請確認是否完整下載")
    
    return len(missing_files) == 0, missing_files

def check_config_files() -> bool:
    """
    檢查配置檔案是否存在
    
    Returns:
        bool: 配置檔案是否完整
    """
    print_subsection("5. 配置檔案檢查")
    
    config_files = [
        'config/config.yaml',
        'requirements.txt',
        'README.md',
        '.gitignore',
    ]
    
    all_exist = True
    
    for config_file in config_files:
        filepath = Path(config_file)
        if filepath.exists():
            print(f"  ✓ {config_file}")
        else:
            print(f"  ❌ {config_file}")
            all_exist = False
    
    if not all_exist:
        print(f"\n  ⚠️  部分配置檔案缺少")
        print(f"  請執行 PowerShell 腳本重新建立")
    
    return all_exist

def check_codebook_generated() -> bool:
    """
    檢查資料字典是否已生成
    
    Returns:
        bool: 資料字典是否存在
    """
    print_subsection("6. 資料字典檢查")
    
    metadata_dir = Path('data/metadata')
    codebook_files = {
        'data_dictionary.md': 'Markdown格式（適合閱讀）',
        'data_dictionary.xlsx': 'Excel格式（適合查詢）',
        'data_dictionary.json': 'JSON格式（適合程式讀取）',
    }
    
    found = 0
    
    for filename, description in codebook_files.items():
        filepath = metadata_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename:30s} ({size_kb:.1f} KB)")
            print(f"    └─ {description}")
            found += 1
        else:
            print(f"  - {filename:30s} (未生成)")
            print(f"    └─ {description}")
    
    if found == 0:
        print(f"\n  ℹ️  資料字典尚未生成")
        print(f"\n  可執行以下程式生成:")
        print(f"  ")
        print(f"  from src.utils.codebook_generator import CodebookGenerator")
        print(f"  generator = CodebookGenerator('data/raw')")
        print(f"  generator.generate_codebook('markdown')")
        print(f"  generator.generate_codebook('excel')")
        return False
    elif found < len(codebook_files):
        print(f"\n  ℹ️  已生成 {found}/{len(codebook_files)} 個格式")
    else:
        print(f"\n  ✓ 所有格式的資料字典已生成")
    
    return True

def print_summary(checks: dict):
    """
    印出檢查結果總結
    
    Parameters:
        checks: 各項檢查的結果字典
    """
    print_section("檢查結果總結")
    
    all_passed = all(checks.values())
    
    print()
    for check_name, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print(" ✅ 所有檢查通過！環境已準備就緒。")
        print("=" * 70)
        print("\n🚀 下一步:")
        print("  1. 開始資料載入:")
        print("     python scripts/01_data_loading.py")
        print()
        print("  2. 或查看完整流程:")
        print("     python scripts/run_all_pipeline.py --help")
        print()
    else:
        print(" ⚠️  部分檢查未通過，請先解決上述問題。")
        print("=" * 70)
        print("\n📝 建議修復步驟:")
        
        if not checks.get('Python版本', True):
            print("  1. 更新Python到3.8以上版本")
        
        if not checks.get('Python套件', True):
            print("  2. 安裝缺少的Python套件:")
            print("     pip install -r requirements.txt")
        
        if not checks.get('資料夾結構', True):
            print("  3. 重新執行資料夾建立腳本:")
            print("     .\\setup_project_structure.ps1")
        
        if not checks.get('資料檔案', True):
            print("  4. 下載資料檔案:")
            print("     https://osf.io/3hvt2/")
            print("     放置於 data/raw/")
        
        print()

def main():
    """主程式"""
    print_section("MIT Moral Machine 專案環境檢查")
    print("\n此腳本將檢查專案執行所需的環境配置")
    print("包含: Python版本、套件、資料夾結構、資料檔案等")
    
    # 執行所有檢查
    checks = {}
    
    checks['Python版本'] = check_python_version()
    checks['Python套件'], missing_packages = check_packages()
    checks['資料夾結構'] = check_directory_structure()
    checks['資料檔案'], missing_files = check_data_files()
    checks['配置檔案'] = check_config_files()
    checks['資料字典'] = check_codebook_generated()
    
    # 印出總結
    print_summary(checks)
    
    # 返回狀態碼
    return 0 if all(checks.values()) else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)