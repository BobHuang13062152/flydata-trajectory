param(
  [Parameter(Mandatory=$true)][string]$RepoName,
  [string]$OrgOrUser = $env:GITHUB_USER,
  [switch]$Private,
  [string]$GhPath
)

# Helper to publish current folder to GitHub as a new repository.
# Requirements: Git & GitHub CLI installed and logged in (gh auth login)

function Resolve-Gh {
  param([string]$GhPath)
  if ($GhPath -and (Test-Path $GhPath)) { return (Resolve-Path $GhPath).Path }
  $cmd = Get-Command gh -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $candidates = @(
    "C:\\Program Files\\GitHub CLI\\gh.exe",
    "C:\\Program Files (x86)\\GitHub CLI\\gh.exe",
    "$env:LOCALAPPDATA\\Programs\\GitHub CLI\\gh.exe"
  )
  foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
  return $null
}

$gh = Resolve-Gh -GhPath $GhPath
if (-not $gh) {
  Write-Error "GitHub CLI 'gh' not found. Install from https://cli.github.com/ or specify -GhPath"
  exit 1
}

if (-not $OrgOrUser) {
  $OrgOrUser = Read-Host "GitHub org/user (e.g. your-username)"
}

$visibility = if ($Private) { 'private' } else { 'public' }

Write-Host "Creating repo $OrgOrUser/$RepoName ($visibility)..."

# Initialize git if needed
if (-not (Test-Path .git)) {
  git init
}

git add .
$status = git status --porcelain
if ($status) {
  git commit -m "Initial commit: publish project" | Out-Null
} else {
  Write-Host "No changes to commit."
}

$remoteUrl = "https://github.com/$OrgOrUser/$RepoName.git"

# Create repo via gh
$repoFull = "$OrgOrUser/$RepoName"
& $gh repo view $repoFull 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  & $gh repo create $repoFull --$visibility --source . --remote origin --push
} else {
  if (-not (git remote | Select-String -SimpleMatch "origin")) {
    git remote add origin $remoteUrl
  }
  git push -u origin HEAD:main
}

Write-Host "Done. Repo URL: https://github.com/$repoFull"
