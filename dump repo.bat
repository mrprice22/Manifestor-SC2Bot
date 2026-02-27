Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
\venv\Scripts\activate
python repo_dump.py --extensions .py .json .yml .ps1
