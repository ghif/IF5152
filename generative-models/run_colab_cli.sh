#!/usr/bin/env bash
# ==============================================================================
# run_colab_cli.sh
# ==============================================================================
# Automated Colab CLI workflow for training Class-Conditional Rectified Flow
# on CIFAR-10 with GPU G4 (or T4/L4) acceleration.
#
# Requirements:
#   - colab CLI installed (e.g. `uv tool install google-colab-cli` or `pip install google-colab-cli`)
#   - Authenticated Google Colab account with GPU access
#
# Usage:
#   ./run_colab_cli.sh [command] [options]
#
# Commands:
#   start     : Provision a new Colab VM session with GPU G4 (or T4/L4)
#   upload    : Upload training script, utils, and dependencies to Colab VM
#   console   : Attach to interactive TTY console on Colab VM
#   train     : Launch training in Colab VM
#   download  : Download checkpoints and generated samples from Colab to local
#   status    : Check active session status and compute unit usage
#   stop      : Terminate remote Colab VM session to prevent wasting compute units
# ==============================================================================

set -euo pipefail

SESSION_NAME="${COLAB_SESSION:-rf-cifar10}"
GPU_TYPE="${COLAB_GPU:-G4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
SAMPLES_DIR="${SCRIPT_DIR}/samples"

COLAB_BIN="$(command -v colab || echo "$HOME/.local/bin/colab")"

if [[ ! -x "$COLAB_BIN" ]]; then
    echo "[ERROR] 'colab' CLI was not found in PATH or ~/.local/bin/colab."
    echo "Install it with:"
    echo "    uv tool install google-colab-cli"
    echo "or:"
    echo "    pip install google-colab-cli"
    exit 1
fi

log_info() {
    echo -e "\033[1;34m[Colab CLI]\033[0m $1"
}

log_success() {
    echo -e "\033[1;32m[Colab CLI]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[Colab CLI]\033[0m $1"
}

cmd_start() {
    log_info "Creating Colab session '${SESSION_NAME}' with accelerator --gpu ${GPU_TYPE}..."
    if ! "$COLAB_BIN" new -s "${SESSION_NAME}" --gpu "${GPU_TYPE}"; then
        log_warn "Failed to create session with --gpu ${GPU_TYPE}. Retrying with fallback --gpu T4..."
        "$COLAB_BIN" new -s "${SESSION_NAME}" --gpu "T4"
    fi
    log_success "Session '${SESSION_NAME}' successfully created!"
    "$COLAB_BIN" status -s "${SESSION_NAME}"
}

cmd_upload() {
    log_info "Uploading project files to Colab session '${SESSION_NAME}'..."
    "$COLAB_BIN" upload -s "${SESSION_NAME}" "${SCRIPT_DIR}/06_rectified_flow_cifar10.py" "/content/06_rectified_flow_cifar10.py"
    "$COLAB_BIN" upload -s "${SESSION_NAME}" "${SCRIPT_DIR}/utils.py" "/content/utils.py"
    if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
        "$COLAB_BIN" upload -s "${SESSION_NAME}" "${SCRIPT_DIR}/requirements.txt" "/content/requirements.txt"
    fi
    log_success "Code files uploaded successfully to /content/!"
}

cmd_console() {
    log_info "Connecting to interactive console for session '${SESSION_NAME}'..."
    "$COLAB_BIN" console -s "${SESSION_NAME}"
}

cmd_ssh() {
    log_info "Connecting to SSH shell for session '${SESSION_NAME}'..."
    "$COLAB_BIN" ssh -s "${SESSION_NAME}"
}

cmd_status() {
    log_info "Querying active Colab sessions and status..."
    "$COLAB_BIN" sessions
    log_info "Session details for '${SESSION_NAME}':"
    "$COLAB_BIN" status -s "${SESSION_NAME}" || true
    log_info "Account compute unit balance:"
    "$COLAB_BIN" usage || true
}

cmd_download() {
    log_info "Downloading training checkpoints and samples from session '${SESSION_NAME}'..."
    mkdir -p "${CHECKPOINT_DIR}" "${SAMPLES_DIR}"

    # Package remote artifacts into a tar archive on the VM and download
    log_info "Archiving remote checkpoints and samples on remote VM..."
    cat << 'EOF' > /tmp/remote_pack.py
import os, tarfile
with tarfile.open("/content/rf_cifar10_artifacts.tar.gz", "w:gz") as tar:
    if os.path.exists("/content/checkpoints"):
        tar.add("/content/checkpoints", arcname="checkpoints")
    if os.path.exists("/content/samples"):
        tar.add("/content/samples", arcname="samples")
print("Archive created successfully.")
EOF
    "$COLAB_BIN" upload -s "${SESSION_NAME}" /tmp/remote_pack.py /content/remote_pack.py
    "$COLAB_BIN" exec -s "${SESSION_NAME}" -f /tmp/remote_pack.py || true
    rm -f /tmp/remote_pack.py

    log_info "Downloading 'rf_cifar10_artifacts.tar.gz' to local machine..."
    mkdir -p /tmp/colab_dl
    if "$COLAB_BIN" download -s "${SESSION_NAME}" /content/rf_cifar10_artifacts.tar.gz /tmp/colab_dl/rf_cifar10_artifacts.tar.gz; then
        log_info "Extracting downloaded artifacts into ${SCRIPT_DIR}..."
        tar -xzf /tmp/colab_dl/rf_cifar10_artifacts.tar.gz -C "${SCRIPT_DIR}"
        rm -rf /tmp/colab_dl
        log_success "Checkpoints and samples successfully synchronized!"
        ls -la "${CHECKPOINT_DIR}"/rectified_flow* 2>/dev/null || true
        ls -la "${SAMPLES_DIR}"/rectified_flow* 2>/dev/null || true
    else
        log_warn "Remote archive download failed or not found. Trying direct file download..."
        "$COLAB_BIN" download -s "${SESSION_NAME}" /content/checkpoints/rectified_flow_cifar10_checkpoint.pt "${CHECKPOINT_DIR}/rectified_flow_cifar10_checkpoint.pt" || true
        "$COLAB_BIN" download -s "${SESSION_NAME}" /content/checkpoints/rectified_flow_cifar10_model.pt "${CHECKPOINT_DIR}/rectified_flow_cifar10_model.pt" || true
        "$COLAB_BIN" download -s "${SESSION_NAME}" /content/checkpoints/rectified_flow_cifar10_model_ema.pt "${CHECKPOINT_DIR}/rectified_flow_cifar10_model_ema.pt" || true
        "$COLAB_BIN" download -s "${SESSION_NAME}" /content/samples/rectified_flow_cifar10_samples.png "${SAMPLES_DIR}/rectified_flow_cifar10_samples.png" || true
    fi
}

cmd_stop() {
    log_info "Stopping Colab session '${SESSION_NAME}'..."
    "$COLAB_BIN" stop -s "${SESSION_NAME}"
    log_success "Session '${SESSION_NAME}' stopped. Compute units preserved."
}

cmd_help() {
    echo "Usage: ./run_colab_cli.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start     Create remote Colab session with GPU G4 (or T4)"
    echo "  upload    Upload python scripts (06_rectified_flow_cifar10.py, utils.py)"
    echo "  console   Attach to TTY console to run/monitor training interactively"
    echo "  ssh       Open SSH connection into Colab VM"
    echo "  download  Download checkpoints and samples to local directories"
    echo "  status    Check active sessions and compute unit usage"
    echo "  stop      Stop Colab session when finished"
    echo ""
    echo "Examples:"
    echo "  ./run_colab_cli.sh start"
    echo "  ./run_colab_cli.sh upload"
    echo "  ./run_colab_cli.sh console"
    echo "    (inside console): python3 06_rectified_flow_cifar10.py --epochs 80 --batch-size 128"
    echo "  ./run_colab_cli.sh download"
    echo "  ./run_colab_cli.sh stop"
}

ACTION="${1:-help}"
shift || true

case "$ACTION" in
    start)
        cmd_start "$@"
        ;;
    upload)
        cmd_upload "$@"
        ;;
    console)
        cmd_console "$@"
        ;;
    ssh)
        cmd_ssh "$@"
        ;;
    status)
        cmd_status "$@"
        ;;
    download)
        cmd_download "$@"
        ;;
    stop)
        cmd_stop "$@"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        echo "Unknown command: $ACTION"
        cmd_help
        exit 1
        ;;
esac
