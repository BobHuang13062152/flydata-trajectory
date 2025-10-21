param(
  [Parameter(Mandatory=$true)][string]$RepoName,
  [string]$OrgOrUser = $env:GITHUB_USER,
  [switch]$Private
)

# Helper to publish current folder to GitHub as a new repository.
# Requirements: gh CLI installed and logged in (gh auth login)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  Write-Error "GitHub CLI 'gh' not found. Install from https://cli.github.com/"
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
git commit -m "Initial commit: publish project"

$remoteUrl = "https://github.com/$OrgOrUser/$RepoName.git"

# Create repo via gh
$repoFull = "$OrgOrUser/$RepoName"
$exists = gh repo view $repoFull 2>$null
if ($LASTEXITCODE -ne 0) {
  gh repo create $repoFull --$visibility --source . --remote origin --push
} else {
  if (-not (git remote | Select-String -SimpleMatch "origin")) {
    git remote add origin $remoteUrl
  }
  git push -u origin HEAD:main
}

Write-Host "Done. Repo URL: https://github.com/$repoFull"
