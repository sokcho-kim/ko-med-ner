# GLiNER2 학습 환경 Docker 이미지 빌드 & 푸시 스크립트
# 사용법: .\build_and_push.ps1 [-Username <dockerhub_username>] [-Tag <tag>]

param(
    [string]$Username = "",
    [string]$Tag = "latest"
)

if (-not $Username) {
    $Username = Read-Host "Docker Hub username"
}

$ImageName = "$Username/gliner2-train"
$FullTag = "${ImageName}:${Tag}"

Write-Host "Building image: $FullTag" -ForegroundColor Cyan

# Build
docker build -t $FullTag -f "$PSScriptRoot/Dockerfile" "$PSScriptRoot"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build succeeded. Pushing to Docker Hub..." -ForegroundColor Green

# Push
docker push $FullTag
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed! Run 'docker login' first." -ForegroundColor Red
    exit 1
}

Write-Host "Pushed: $FullTag" -ForegroundColor Green
