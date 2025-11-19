# Manual CIFAR-10 Dataset Download Script
# Use this if automatic download fails

$dataDir = "data"
$cifar10Url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
$outputFile = "$dataDir\cifar-10-python.tar.gz"

Write-Host "=== CIFAR-10 Manual Download ===" -ForegroundColor Cyan
Write-Host ""

# Create data directory
if (!(Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir
    Write-Host "Created directory: $dataDir" -ForegroundColor Green
}

# Download the dataset
Write-Host "Downloading CIFAR-10 from: $cifar10Url" -ForegroundColor Yellow
Write-Host "This may take a few minutes (170 MB)..." -ForegroundColor Yellow
Write-Host ""

try {
    # Use Invoke-WebRequest with progress
    $ProgressPreference = 'Continue'
    Invoke-WebRequest -Uri $cifar10Url -OutFile $outputFile -UseBasicParsing
    Write-Host "`nDownload completed successfully!" -ForegroundColor Green
    
    # Extract the archive
    Write-Host "Extracting archive..." -ForegroundColor Yellow
    
    # Use tar if available (Windows 10+)
    if (Get-Command tar -ErrorAction SilentlyContinue) {
        tar -xzf $outputFile -C $dataDir
        Write-Host "Extraction completed!" -ForegroundColor Green
    } else {
        Write-Host "Please extract the file manually:" -ForegroundColor Yellow
        Write-Host "  File location: $outputFile" -ForegroundColor White
        Write-Host "  Extract to: $dataDir" -ForegroundColor White
    }
    
    Write-Host "`nDataset is ready! You can now run training." -ForegroundColor Green
} catch {
    Write-Host "`nDownload failed: $_" -ForegroundColor Red
    Write-Host "`nAlternative: Download manually from:" -ForegroundColor Yellow
    Write-Host "  $cifar10Url" -ForegroundColor White
    Write-Host "And save it to: $outputFile" -ForegroundColor White
}
