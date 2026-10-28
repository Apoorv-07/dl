param(
    [Parameter(Mandatory = $true)]
    [string]$Member2Package,
    [string]$CheckpointPath = "..\member2\checkpoints_validation\best_model.pt"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not [System.IO.Path]::IsPathRooted($Member2Package)) {
    $Member2Package = Join-Path $ProjectRoot $Member2Package
}
$Member2Package = (Resolve-Path $Member2Package).Path
if (-not [System.IO.Path]::IsPathRooted($CheckpointPath)) {
    $CheckpointPath = Join-Path $ProjectRoot $CheckpointPath
}
$CheckpointPath = (Resolve-Path $CheckpointPath).Path
$Destination = Join-Path $ProjectRoot "member2_artifacts"
New-Item -ItemType Directory -Path $Destination -Force | Out-Null

$Required = @("member2_records.npz", "model_config.yaml", "feature_schema.json", "preprocessing.json")
foreach ($Name in $Required) {
    $Source = Join-Path $Member2Package $Name
    if (-not (Test-Path $Source -PathType Leaf)) { throw "Required package artifact missing: $Source" }
    Copy-Item $Source (Join-Path $Destination $Name) -Force
}
Copy-Item $CheckpointPath (Join-Path $Destination "checkpoint.pt") -Force
$OptionalPairs = Join-Path $Member2Package "sample_pairs.pt"
if (Test-Path $OptionalPairs -PathType Leaf) {
    Copy-Item $OptionalPairs (Join-Path $Destination "sample_pairs.pt") -Force
}
Write-Host "Copied Member-2 artifacts to $Destination"
Write-Host "Run from the member3 project directory: python tools/smoke_test_member2_adapter.py --pairs member2_artifacts/sample_pairs.pt"
Write-Host "Do not git-add checkpoint.pt, *.npz, *.pt, or *.csv; the nested .gitignore excludes them."
