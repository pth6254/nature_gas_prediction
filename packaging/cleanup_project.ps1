param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$expectedRoot = 'C:\Users\Laptop PC\Desktop\nature_gas_prediction'
if ($projectRoot -ne $expectedRoot) { throw "Unexpected workspace: $projectRoot" }
$backupRoot = 'C:\Users\Laptop PC\Desktop\nature_gas_prediction_backup_20260906_cleanup'
if ([IO.Path]::GetDirectoryName($backupRoot) -ne [IO.Path]::GetDirectoryName($projectRoot)) { throw 'Unexpected backup parent' }
if ($backupRoot -eq $projectRoot -or $backupRoot.StartsWith($projectRoot + '\')) { throw 'Backup must be outside project' }
$relativeTargets = @(
    'README.md', 'REPORT_DRAFT.md', 'REPORT_before_final_v1.md',
    'submission_as_provided.csv', 'outputs_as_provided', 'figures_as_provided', 'figures',
    'notebooks\01_eda.ipynb', 'presentation\before_rubric_revision',
    'tmp', '.mplconfig', 'pytest-cache-files-a64omk0d', 'pytest-cache-files-f8bg2_b9',
    'src\__pycache__', 'tests\__pycache__',
    'outputs_stability\cache', 'outputs_variable_suitability\cache',
    'outputs_incremental_fundamentals\cache', 'deliverables'
)
# Preserve feature-audit warning JSON: its report reader still consumes these.
$featureCache = Join-Path $projectRoot 'outputs_feature_audit\cache'
if (Test-Path -LiteralPath $featureCache) {
    Get-ChildItem -LiteralPath $featureCache -File -Filter '*.csv' | ForEach-Object {
        $relativeTargets += $_.FullName.Substring($projectRoot.Length + 1)
    }
}
$plan = foreach ($relative in $relativeTargets) {
    $source = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    $destination = [IO.Path]::GetFullPath((Join-Path $backupRoot $relative))
    if (-not $source.StartsWith($projectRoot + '\') -or -not $destination.StartsWith($backupRoot + '\')) { throw 'Path escaped expected boundary' }
    if (-not (Test-Path -LiteralPath $source)) { continue }
    $resolved = (Resolve-Path -LiteralPath $source).Path
    if ($resolved -ne $source) { throw "Unexpected resolved path: $resolved" }
    $item = Get-Item -LiteralPath $source -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing reparse point: $source" }
    [pscustomobject]@{relative=$relative; source=$source; backup=$destination; isDirectory=$item.PSIsContainer}
}
if (-not $Apply) { $plan | ConvertTo-Json -Depth 3; exit }
if (Test-Path -LiteralPath $backupRoot) { throw 'Backup already exists; refusing to overwrite' }
New-Item -ItemType Directory -Path $backupRoot | Out-Null
$plan | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $backupRoot 'move_plan.json') -Encoding UTF8
$completed = @()
foreach ($entry in $plan) {
    $parent = [IO.Path]::GetDirectoryName($entry.backup)
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -LiteralPath $entry.source -Destination $entry.backup
    if ((Test-Path -LiteralPath $entry.source) -or -not (Test-Path -LiteralPath $entry.backup)) { throw "Move verification failed: $($entry.relative)" }
    $completed += $entry
    $completed | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $backupRoot 'completed_moves.json') -Encoding UTF8
}
Write-Output "Moved $($completed.Count) scoped entries. Backup: $backupRoot"
