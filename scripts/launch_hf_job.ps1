#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Launches the CrisisOps GRPO training job on HuggingFace Jobs (A100 80GB).

.DESCRIPTION
  Wraps `hf jobs run` with the right Docker image, secrets, env vars, and
  entrypoint. The job clones the GitHub repo at start, installs the GRPO
  training stack, runs Qwen3-8B + Unsloth + TRL GRPO, and pushes artifacts
  to the HF Hub model repo `HF_OUTPUT_REPO`.

.PARAMETER Flavor
  HF Jobs hardware flavor (default: a100-large, $2.50/hr).

.PARAMETER Timeout
  Max wall clock the job may run (default: 6h, ~$15 worst case).

.PARAMETER OutputRepo
  HF Hub model repo to push artifacts/adapter to.

.PARAMETER MaxGrpoSteps
  Override the GRPO step budget. Default 300.

.PARAMETER NumTrainEpisodes
  Override the prompt curriculum size. Default 360.

.PARAMETER WandbKey
  Optional W&B API key. If supplied, training also logs to W&B.

.EXAMPLE
  pwsh ./scripts/launch_hf_job.ps1
  pwsh ./scripts/launch_hf_job.ps1 -Timeout 5h -MaxGrpoSteps 200
  pwsh ./scripts/launch_hf_job.ps1 -WandbKey $env:WANDB_API_KEY

.NOTES
  Requires `hf` CLI (huggingface_hub >= 0.26) authenticated as a user with a
  payment method enabled (Hugging Face Jobs is paid compute).
#>

[CmdletBinding()]
param(
    [string]$Flavor = "a100-large",
    [string]$Timeout = "6h",
    [string]$OutputRepo = "Vk224/crisisops-qwen3-8b-grpo",
    [string]$RepoUrl = "https://github.com/Vk2245/CrisisOps-Multi-Agent-SRE-Training-via-OpenEnv.git",
    [string]$RepoRef = "main",
    [int]$MaxGrpoSteps = 300,
    [int]$NumTrainEpisodes = 360,
    [int]$MaxSeqLength = 4096,
    [int]$LoraRank = 32,
    [int]$PerDeviceTrainBatchSize = 4,
    [int]$GradientAccumulationSteps = 2,
    [int]$NumGenerations = 4,
    [int]$MaxPromptLength = 2048,
    [int]$MaxCompletionLength = 1024,
    [double]$ModelGpuMemoryUtilization = 0.70,
    [double]$VllmGpuMemoryUtilization = 0.35,
    [bool]$FastInference = $true,
    [bool]$UseVllm = $true,
    [string]$WandbKey = "",
    [string]$BaseImage = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " CrisisOps GRPO Training - HuggingFace Jobs Launcher" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Flavor:          $Flavor"
Write-Host " Timeout:         $Timeout"
Write-Host " Base image:      $BaseImage"
Write-Host " Repo:            $RepoUrl @ $RepoRef"
Write-Host " Output repo:     $OutputRepo"
Write-Host " GRPO steps:      $MaxGrpoSteps"
Write-Host " Episodes:        $NumTrainEpisodes"
Write-Host " Max seq length:  $MaxSeqLength"
Write-Host " LoRA rank:       $LoraRank"
Write-Host " Batch/gen:       batch=$PerDeviceTrainBatchSize grad_accum=$GradientAccumulationSteps generations=$NumGenerations"
Write-Host " vLLM/model mem:  vllm=$VllmGpuMemoryUtilization model=$ModelGpuMemoryUtilization"
Write-Host " Fast/vLLM:       fast_inference=$FastInference use_vllm=$UseVllm"
Write-Host " W&B logging:     $([bool]$WandbKey)"
Write-Host "=================================================================="

# Quick sanity check: hf CLI must be present and authenticated.
$hfWho = & hf auth whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "hf CLI is not authenticated. Run: hf auth login"
    exit 1
}
Write-Host "Authenticated HF user: $hfWho"

# Bash command that the container will run. We download the entrypoint
# directly from the repo so we can iterate on it without rebuilding the
# Docker image.
$entrypointUrl = "https://raw.githubusercontent.com/Vk2245/CrisisOps-Multi-Agent-SRE-Training-via-OpenEnv/$RepoRef/scripts/hf_job_entrypoint.sh"
$bashCommand = @"
set -e
apt-get update -qq && apt-get install -y -qq curl git
curl -fsSL "$entrypointUrl" -o /tmp/entrypoint.sh
chmod +x /tmp/entrypoint.sh
exec /tmp/entrypoint.sh
"@

$arguments = @(
    "jobs", "run",
    "--flavor", $Flavor,
    "--timeout", $Timeout,
    "--secrets", "HF_TOKEN",
    "--env", "REPO_URL=$RepoUrl",
    "--env", "REPO_REF=$RepoRef",
    "--env", "HF_OUTPUT_REPO=$OutputRepo",
    "--env", "MAX_GRPO_STEPS=$MaxGrpoSteps",
    "--env", "NUM_TRAIN_EPISODES=$NumTrainEpisodes",
    "--env", "MAX_SEQ_LENGTH=$MaxSeqLength",
    "--env", "LORA_RANK=$LoraRank",
    "--env", "PER_DEVICE_TRAIN_BATCH_SIZE=$PerDeviceTrainBatchSize",
    "--env", "GRADIENT_ACCUMULATION_STEPS=$GradientAccumulationSteps",
    "--env", "NUM_GENERATIONS=$NumGenerations",
    "--env", "MAX_PROMPT_LENGTH=$MaxPromptLength",
    "--env", "MAX_COMPLETION_LENGTH=$MaxCompletionLength",
    "--env", "MODEL_GPU_MEMORY_UTILIZATION=$ModelGpuMemoryUtilization",
    "--env", "VLLM_GPU_MEMORY_UTILIZATION=$VllmGpuMemoryUtilization",
    "--env", "FAST_INFERENCE=$FastInference",
    "--env", "USE_VLLM=$UseVllm",
    "--env", "MODEL_NAME=unsloth/Qwen3-8B"
)

if ($WandbKey) {
    $env:WANDB_API_KEY = $WandbKey
    $arguments += @("--secrets", "WANDB_API_KEY")
}

$arguments += @(
    "--detach",
    $BaseImage,
    "bash", "-c", $bashCommand
)

Write-Host ""
Write-Host "Launching: hf $($arguments -join ' ')" -ForegroundColor Yellow
Write-Host ""

& hf @arguments
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Error "hf jobs run exited with code $exitCode"
    exit $exitCode
}

Write-Host ""
Write-Host "Job submitted. Useful commands:" -ForegroundColor Green
Write-Host "  hf jobs ps                 # list running/recent jobs"
Write-Host "  hf jobs logs <JOB_ID>      # stream stdout/stderr"
Write-Host "  hf jobs inspect <JOB_ID>   # job details"
Write-Host "  hf jobs cancel <JOB_ID>    # stop the job"
Write-Host ""
Write-Host "Once the job finishes, artifacts will appear at:" -ForegroundColor Green
Write-Host "  https://huggingface.co/$OutputRepo"
