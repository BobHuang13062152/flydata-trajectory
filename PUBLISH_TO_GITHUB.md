# Publish this project to GitHub (Windows PowerShell)

## 1) Initialize git (first time only)
Run these from the project root `C:\NCHC_DATA\flydata`.

```powershell
Set-Location -Path 'C:\NCHC_DATA\flydata'
git init
# Review what will be added (respects .gitignore)
git status --short
# Stage and commit
git add .
git commit -m "Initial commit: aviation trajectory retrieval & prediction system (minimal runtime)"
```

## 2) Create a new GitHub repository
- Using GitHub CLI (recommended):
```powershell
gh repo create <your-user-or-org>/<repo-name> --public --source . --remote origin --push
```
- Or manually via GitHub UI:
  1. Create an empty repo named `<repo-name>`
  2. Then add remote and push:
```powershell
git remote add origin https://github.com/<your-user-or-org>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 3) After publish
- Add screenshots to `README.md` (e.g., `paper/figures/ui_demo_top5.png`).
- Open issues for: evaluation script + table, packaging, CI, etc.

## Notes
- This repository intentionally includes only actually used runtime assets:
  - Include: `flight_prediction_server_fixed.py`, `demo_with_real_data_fixed.html`, `openflights_adapter.py`, `requirements.txt`, `models/README.md`, `README.md`, `start_server.bat`, `.gitignore`, `.gitattributes`.
  - Exclude (via .gitignore): large data (`flights_*.geojson`), trained weights (`models/*.pt`), papers, notebooks, diagnostics, and experimental scripts not required at runtime.
- Keep secrets out of git. Use repository secrets if adding CI/CD.
