@echo off
setlocal
pushd "%~dp0"
uv run --extra data --extra retrieval python processor.py %*
set "preprocess_exit_code=%ERRORLEVEL%"
popd
exit /b %preprocess_exit_code%
