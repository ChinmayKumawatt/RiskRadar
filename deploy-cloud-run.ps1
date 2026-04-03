param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [string]$ServiceName = "riskradar",

    [string]$Region = "us-central1",

    [string]$Memory = "1Gi",

    [string]$Cpu = "1"
)

$ErrorActionPreference = "Stop"

$gcloud = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

if (-not (Test-Path $gcloud)) {
    throw "gcloud was not found at $gcloud"
}

Write-Host "Setting active project to $ProjectId ..."
& $gcloud config set project $ProjectId

Write-Host "Enabling required Cloud APIs ..."
& $gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

Write-Host "Deploying $ServiceName to Cloud Run in $Region ..."
& $gcloud run deploy $ServiceName `
    --source . `
    --project $ProjectId `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --memory $Memory `
    --cpu $Cpu `
    --min-instances 0 `
    --max-instances 2
