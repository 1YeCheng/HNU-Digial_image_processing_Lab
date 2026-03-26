@echo off
set /p folder="Enter the folder name you want to push (e.g., 20260326_work2): "

echo.
echo --- Starting Git Process for %folder% ---

:: 1. 只添加指定的文件夹
git add %folder%/*

:: 2. 同时更新 README（通常这个也需要同步）
git add README.md

:: 3. 执行提交
git commit -m "feat: update %folder% and readme"

:: 4. 推送到远程
git push origin master

echo --- Push Completed! ---
pause