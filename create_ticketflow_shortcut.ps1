$ErrorActionPreference = "Stop"

$projectRoot = "D:\Study\shixi\ticketflow"
$target = Join-Path $projectRoot "start_ticketflow_demo.bat"
$desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
$shortcutName = "TicketFlow " + [char]0x6F14 + [char]0x793A + [char]0x7248 + ".lnk"
$shortcutPath = Join-Path $desktop $shortcutName
$temporaryShortcutPath = Join-Path $desktop "TicketFlow_Demo.lnk"

if (-not (Test-Path -LiteralPath $target)) {
    throw "启动脚本不存在：$target"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($temporaryShortcutPath)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = "TicketFlow demo launcher"
$shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,220"
$shortcut.Save()

if (Test-Path -LiteralPath $shortcutPath) {
    Remove-Item -LiteralPath $shortcutPath -Force
}
Move-Item -LiteralPath $temporaryShortcutPath -Destination $shortcutPath -Force

Write-Host "已创建桌面快捷方式：$shortcutPath"
