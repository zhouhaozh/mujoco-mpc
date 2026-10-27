#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

SKIP_SYSTEM_DEPS=0
JOBS=2
INDEX_URL="https://pypi.org/simple"

usage() {
    cat <<'EOF'
用法：bash tools/install_ubuntu.sh [选项]

在 Ubuntu 24.04 x86_64 上安装仿真依赖、编译工程、运行测试并执行无界面仿真冒烟测试。
脚本可从任意工作目录执行，默认使用 2 个并行编译任务。

选项：
  --skip-system-deps  跳过 apt 系统依赖检查与安装
  --jobs N           并行编译任务数，N 必须是正整数（默认：2）
  --index-url URL     Python 包索引地址（默认：https://pypi.org/simple）
  --help              显示本帮助并退出
EOF
}

error() {
    printf '错误：%s\n' "$*" >&2
}

while (($# > 0)); do
    case "$1" in
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=1
            shift
            ;;
        --jobs)
            if (($# < 2)); then
                error "--jobs 需要一个正整数参数"
                usage >&2
                exit 2
            fi
            JOBS="$2"
            shift 2
            ;;
        --index-url)
            if (($# < 2)) || [[ -z "$2" ]]; then
                error "--index-url 需要一个非空 URL"
                usage >&2
                exit 2
            fi
            INDEX_URL="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "未知参数：$1"
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    error "--jobs 必须是正整数，当前值：${JOBS}"
    exit 2
fi

ARCHITECTURE="$(uname -m)"
if [[ "$ARCHITECTURE" != "x86_64" ]]; then
    error "仅支持 x86_64 架构，当前架构：${ARCHITECTURE}"
    exit 1
fi

if [[ ! -r /etc/os-release ]]; then
    error "无法读取 /etc/os-release；仅支持 Ubuntu 24.04"
    exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    error "仅支持 Ubuntu 24.04，当前系统：${PRETTY_NAME:-未知}"
    exit 1
fi

SYSTEM_PACKAGES=(
    python3
    python3-venv
    build-essential
    gcc-13
    g++-13
    cmake
    libeigen3-dev
    nlohmann-json3-dev
    libglfw3
    libgl1-mesa-dev
)

if ((SKIP_SYSTEM_DEPS == 1)); then
    printf '已跳过 apt 系统依赖检查。\n'
else
    if ! command -v dpkg-query >/dev/null 2>&1; then
        error "找不到 dpkg-query，无法检查系统依赖"
        exit 1
    fi

    MISSING_PACKAGES=()
    for package in "${SYSTEM_PACKAGES[@]}"; do
        status="$(dpkg-query -W '-f=${Status}' "$package" 2>/dev/null || true)"
        if [[ "$status" != "install ok installed" ]]; then
            MISSING_PACKAGES+=("$package")
        fi
    done

    if ((${#MISSING_PACKAGES[@]} == 0)); then
        printf '系统依赖已全部安装，跳过 apt-get。\n'
    else
        printf '将安装缺少的系统依赖：%s\n' "${MISSING_PACKAGES[*]}"
        if ((EUID == 0)); then
            APT_GET=(apt-get)
        else
            if ! command -v sudo >/dev/null 2>&1; then
                error "安装系统依赖需要 sudo，但当前 PATH 中找不到 sudo"
                exit 1
            fi
            APT_GET=(sudo apt-get)
        fi
        "${APT_GET[@]}" update
        "${APT_GET[@]}" install -y "${MISSING_PACKAGES[@]}"
    fi
fi

printf '开始创建 Python 环境并编译工程（并行任务数：%s）。\n' "$JOBS"
python3 "${PROJECT_ROOT}/tools/setup_predictor_ubuntu.py" \
    --build-dir sckd_sim/build \
    --jobs "$JOBS" \
    --index-url "$INDEX_URL"

printf '开始运行无界面仿真冒烟测试。\n'
(
    cd "${PROJECT_ROOT}/sckd_sim"
    unset M2_VIEW
    ./build/m2_intercept models/generated/x4_tennis_demo.mjcf 3 qp 0.25 2026 pfly
)
printf '安装、编译和仿真冒烟测试均已完成。\n'
