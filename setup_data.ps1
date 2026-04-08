<#
.SYNOPSIS
    Download XAUUSD H1 + M15 data via dukascopy-node and merge into single CSVs.
    Run this on a new server before executing backtest scripts.

.PREREQUISITES
    - Node.js (v14+)
    - npm install -g dukascopy-node   OR just use npx (auto-downloads)

.USAGE
    powershell -ExecutionPolicy Bypass -File setup_data.ps1
#>

$ErrorActionPreference = "Continue"

$targetDir = "data/download"
$tempH1Dir = "data/download/_tmp_h1"
$tempM15Dir = "data/download/_tmp_m15"

$h1Output = "$targetDir/xauusd-h1-bid-2015-01-01-2026-03-25.csv"
$m15Output = "$targetDir/xauusd-m15-bid-2015-01-01-2026-03-25.csv"

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempH1Dir | Out-Null
New-Item -ItemType Directory -Force -Path $tempM15Dir | Out-Null

# ═══════════════════════════════════════════════════════════════
# Check prerequisites
# ═══════════════════════════════════════════════════════════════
Write-Host "=== XAUUSD Data Setup ===" -ForegroundColor Cyan
Write-Host ""

$nodeVer = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Node.js not found. Install from https://nodejs.org/" -ForegroundColor Red
    exit 1
}
Write-Host "Node.js: $nodeVer" -ForegroundColor Green

# ═══════════════════════════════════════════════════════════════
# Download chunks - 2-year segments to avoid timeout
# ═══════════════════════════════════════════════════════════════

$yearChunks = @(
    @("2015-01-01", "2017-01-01"),
    @("2017-01-01", "2019-01-01"),
    @("2019-01-01", "2021-01-01"),
    @("2021-01-01", "2023-01-01"),
    @("2023-01-01", "2025-01-01"),
    @("2025-01-01", "2026-04-06")
)

# M15 uses quarterly chunks (dukascopy has stricter limits on M15)
$m15Chunks = @(
    @("2015-01-01", "2015-07-01"), @("2015-07-01", "2016-01-01"),
    @("2016-01-01", "2016-07-01"), @("2016-07-01", "2017-01-01"),
    @("2017-01-01", "2017-07-01"), @("2017-07-01", "2018-01-01"),
    @("2018-01-01", "2018-07-01"), @("2018-07-01", "2019-01-01"),
    @("2019-01-01", "2019-07-01"), @("2019-07-01", "2020-01-01"),
    @("2020-01-01", "2020-07-01"), @("2020-07-01", "2021-01-01"),
    @("2021-01-01", "2021-07-01"), @("2021-07-01", "2022-01-01"),
    @("2022-01-01", "2022-07-01"), @("2022-07-01", "2023-01-01"),
    @("2023-01-01", "2023-07-01"), @("2023-07-01", "2024-01-01"),
    @("2024-01-01", "2024-07-01"), @("2024-07-01", "2025-01-01"),
    @("2025-01-01", "2025-07-01"), @("2025-07-01", "2026-04-06")
)

function Download-Chunks {
    param(
        [array]$chunks,
        [string]$timeframe,
        [string]$outDir
    )

    $total = $chunks.Count
    $done = 0
    $failed = @()

    Write-Host "`n--- Downloading XAUUSD $timeframe ($total chunks) ---" -ForegroundColor Yellow

    foreach ($chunk in $chunks) {
        $from = $chunk[0]
        $to = $chunk[1]
        $done++
        Write-Host "  [$done/$total] $from -> $to ... " -NoNewline

        $output = npx dukascopy-node -i xauusd -from $from -to $to -t $timeframe -p bid -f csv -v -dir $outDir 2>&1
        $outputStr = $output -join "`n"

        if ($outputStr -match "File saved") {
            Write-Host "OK" -ForegroundColor Green
        } else {
            Write-Host "FAIL" -ForegroundColor Red
            $failed += "$from -> $to"
            Write-Host "    Retrying in 5s... " -NoNewline
            Start-Sleep -Seconds 5
            $output = npx dukascopy-node -i xauusd -from $from -to $to -t $timeframe -p bid -f csv -v -dir $outDir 2>&1
            $outputStr = $output -join "`n"
            if ($outputStr -match "File saved") {
                Write-Host "OK (retry)" -ForegroundColor Yellow
                $failed = $failed | Where-Object { $_ -ne "$from -> $to" }
            } else {
                Write-Host "FAIL (retry)" -ForegroundColor Red
            }
        }
    }

    if ($failed.Count -gt 0) {
        Write-Host "  Failed chunks:" -ForegroundColor Red
        $failed | ForEach-Object { Write-Host "    $_" }
    }
    return $failed.Count
}

# ═══════════════════════════════════════════════════════════════
# Step 1: Download H1
# ═══════════════════════════════════════════════════════════════
$h1Fails = Download-Chunks -chunks $yearChunks -timeframe "h1" -outDir $tempH1Dir

# ═══════════════════════════════════════════════════════════════
# Step 2: Download M15
# ═══════════════════════════════════════════════════════════════
$m15Fails = Download-Chunks -chunks $m15Chunks -timeframe "m15" -outDir $tempM15Dir

# ═══════════════════════════════════════════════════════════════
# Step 3: Merge CSVs
# ═══════════════════════════════════════════════════════════════

function Merge-CSVFiles {
    param(
        [string]$sourceDir,
        [string]$outputFile
    )

    Write-Host "`n  Merging -> $outputFile ... " -NoNewline

    $csvFiles = Get-ChildItem $sourceDir -Filter "*.csv" | Sort-Object Name
    if ($csvFiles.Count -eq 0) {
        Write-Host "SKIP (no files)" -ForegroundColor Red
        return
    }

    $headerWritten = $false
    $lineCount = 0

    foreach ($file in $csvFiles) {
        $lines = Get-Content $file.FullName
        if (-not $headerWritten) {
            $lines | Set-Content $outputFile -Encoding UTF8
            $headerWritten = $true
            $lineCount += $lines.Count
        } else {
            # Skip header (first line)
            $dataLines = $lines | Select-Object -Skip 1
            $dataLines | Add-Content $outputFile -Encoding UTF8
            $lineCount += $dataLines.Count
        }
    }

    $sizeMB = [math]::Round((Get-Item $outputFile).Length / 1MB, 1)
    Write-Host "OK ($lineCount lines, ${sizeMB} MB)" -ForegroundColor Green
}

Write-Host "`n--- Merging downloaded chunks ---" -ForegroundColor Yellow
Merge-CSVFiles -sourceDir $tempH1Dir -outputFile $h1Output
Merge-CSVFiles -sourceDir $tempM15Dir -outputFile $m15Output

# ═══════════════════════════════════════════════════════════════
# Step 4: Verify
# ═══════════════════════════════════════════════════════════════
Write-Host "`n--- Verification ---" -ForegroundColor Yellow

foreach ($f in @($h1Output, $m15Output)) {
    if (Test-Path $f) {
        $info = Get-Item $f
        $sizeMB = [math]::Round($info.Length / 1MB, 1)
        $lines = (Get-Content $f | Measure-Object -Line).Lines
        Write-Host "  $($info.Name): ${sizeMB} MB, $lines lines" -ForegroundColor Green
    } else {
        Write-Host "  MISSING: $f" -ForegroundColor Red
    }
}

# ═══════════════════════════════════════════════════════════════
# Cleanup temp dirs
# ═══════════════════════════════════════════════════════════════
Write-Host "`nClean up temp dirs? (y/n): " -NoNewline
$ans = Read-Host
if ($ans -eq "y") {
    Remove-Item -Recurse -Force $tempH1Dir
    Remove-Item -Recurse -Force $tempM15Dir
    Write-Host "Temp dirs removed." -ForegroundColor Green
} else {
    Write-Host "Temp dirs kept at: $tempH1Dir, $tempM15Dir"
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
if ($h1Fails -gt 0 -or $m15Fails -gt 0) {
    Write-Host "WARNING: Some chunks failed. Re-run the script or download manually." -ForegroundColor Yellow
} else {
    Write-Host "All data ready. You can now run the backtest scripts." -ForegroundColor Green
}
