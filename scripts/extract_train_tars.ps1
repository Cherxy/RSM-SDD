param(
    [string]$Root = 'E:\LCX\datasets\train',
    [string]$LogPath = (Join-Path $PSScriptRoot 'extract_train_tars.log')
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)

    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $LogPath -Value $line
    Write-Output $line
}

function Get-DoneNames {
    param([string]$Path)

    $done = New-Object 'System.Collections.Generic.HashSet[string]'
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $done
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match ' DONE \[\d+/\d+\] (?<name>\S+) seconds=') {
            [void]$done.Add($Matches['name'])
        }
    }

    return $done
}

if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "Root directory does not exist: $Root"
}

$archives = @(Get-ChildItem -LiteralPath $Root -File -Filter '*.tar' | Sort-Object Name)
$total = $archives.Count
$doneNames = Get-DoneNames -Path $LogPath
$failed = New-Object System.Collections.Generic.List[string]
$skipped = 0
$extracted = 0

Write-Log "START root=$Root count=$total skippedKnown=$($doneNames.Count) mode=single-process"

for ($i = 0; $i -lt $total; $i++) {
    $archive = $archives[$i]
    $index = $i + 1
    $name = [System.IO.Path]::GetFileNameWithoutExtension($archive.Name)

    if ($doneNames.Contains($name)) {
        $skipped++
        continue
    }

    $destination = Join-Path $archive.DirectoryName $name
    if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
        New-Item -ItemType Directory -Path $destination | Out-Null
    }

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Log "BEGIN [$index/$total] $($archive.FullName) -> $destination"

    $tarOutput = & tar -xf $archive.FullName -C $destination 2>&1
    $exitCode = $LASTEXITCODE
    $timer.Stop()

    foreach ($line in $tarOutput) {
        Write-Log "TAR [$index/$total] $line"
    }

    if ($exitCode -eq 0) {
        $extracted++
        Write-Log "DONE [$index/$total] $name seconds=$([math]::Round($timer.Elapsed.TotalSeconds, 2))"
    }
    else {
        $failed.Add($archive.FullName)
        Write-Log "FAIL [$index/$total] $name exit=$exitCode seconds=$([math]::Round($timer.Elapsed.TotalSeconds, 2))"
    }
}

if ($failed.Count -gt 0) {
    Write-Log "END failed=$($failed.Count) extracted=$extracted skipped=$skipped total=$total"
    foreach ($item in $failed) {
        Write-Log "FAILED_ARCHIVE $item"
    }
    exit 1
}

Write-Log "END success=$($extracted + $skipped) extracted=$extracted skipped=$skipped total=$total"
exit 0
