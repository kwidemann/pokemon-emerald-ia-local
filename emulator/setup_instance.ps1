# Prepara una instancia de mGBA ya abierta: carga el script Lua y pone el avance rapido sin limite.
# mGBA 0.10.5 no tiene opcion de linea de comandos para scripts, asi que se manejan sus menus con UI Automation.
# Lo llama `scripts/launch_emulators.py`; se puede usar suelto:
#   powershell -NoProfile -File emulator\setup_instance.ps1 -ProcessId 1234 -ScriptPath C:\ruta\server.lua
#
# Notas (probadas en vivo, ver specs.md 7.6):
#  - Mientras el dialogo "Select script to load" esta abierto, cualquier llamada de UI Automation se cuelga: el clic en
#    "Load script..." va en un proceso aparte y no se espera.
#  - El avance rapido se activa con InvokePattern; TogglePattern.Toggle() no tuvo efecto.
#  - Los menus salen en el idioma de Windows; se prueban los nombres en espanol e ingles.
#  - Este archivo es solo ASCII a proposito (PowerShell 5.1 lee mal UTF-8 sin BOM): los acentos van con [char].
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$ScriptPath
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms, UIAutomationClient, UIAutomationTypes
Add-Type @"
using System; using System.Text; using System.Runtime.InteropServices;
public class Win32Fg {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowEx(IntPtr parent, IntPtr after, string cls, string title);
  [DllImport("user32.dll")] public static extern IntPtr GetDlgItem(IntPtr dlg, int id);
  [DllImport("user32.dll", EntryPoint = "SendMessageW", CharSet = CharSet.Unicode)] public static extern IntPtr SendText(IntPtr h, int msg, IntPtr w, string l);
  [DllImport("user32.dll", EntryPoint = "SendMessageW")] public static extern IntPtr SendMsg(IntPtr h, int msg, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr h);
}
"@

$AE = [System.Windows.Automation.AutomationElement]
$TS = [System.Windows.Automation.TreeScope]
$MENUITEM = [System.Windows.Automation.ControlType]::MenuItem

# Nombres de los menus (es, en)
$tools = @("Herramientas", "Tools")
$scripting = @("Scripting...")
$emulation = @(("Emulaci" + [char]0xF3 + "n"), "Emulation")
$fastForward = @(("Avance r" + [char]0xE1 + "pido"), "Fast forward")
$fastSpeed = @(("Velocidad de avance r" + [char]0xE1 + "pido"), "Fast forward speed")
$unbounded = @(("Sin l" + [char]0xED + "mite"), "Unbounded")
$fileMenu = @("File", "Archivo")
$loadScript = @("Load script...", "Cargar script...")

function Get-Windows {
    $c = New-Object System.Windows.Automation.PropertyCondition($AE::ProcessIdProperty, $ProcessId)
    @($AE::RootElement.FindAll($TS::Children, $c))
}
function Get-MainWindow { Get-Windows | Where-Object { $_.Current.Name -like "mGBA*" } | Select-Object -First 1 }
function Get-ScriptWindow { Get-Windows | Where-Object { $_.Current.Name -eq "Scripting" } | Select-Object -First 1 }
function Find-Item($scope, [string[]]$names) {
    foreach ($n in $names) {
        $c = New-Object System.Windows.Automation.AndCondition(
            (New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, $MENUITEM)),
            (New-Object System.Windows.Automation.PropertyCondition($AE::NameProperty, $n)))
        $f = $scope.FindFirst($TS::Descendants, $c)
        if ($f) { return $f }
    }
    return $null
}
function Expand-Menu($scope, [string[]]$names) {
    $m = Find-Item $scope $names
    if (-not $m) { throw "no encuentro el menu $($names -join '/')" }
    $m.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand()
    Start-Sleep -Milliseconds 600
}
function Invoke-MenuItem($scope, [string[]]$names) {
    $m = Find-Item $scope $names
    if (-not $m) { throw "no encuentro el elemento $($names -join '/')" }
    $m.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 600
}
function Foreground-IsOurDialog {
    $h = [Win32Fg]::GetForegroundWindow()
    $fgPid = 0; [void][Win32Fg]::GetWindowThreadProcessId($h, [ref]$fgPid)
    $cls = New-Object System.Text.StringBuilder 256; [void][Win32Fg]::GetClassName($h, $cls, 256)
    return ($fgPid -eq $ProcessId -and $cls.ToString() -eq "#32770")
}
function Close-Menus {
    $h = [Win32Fg]::GetForegroundWindow()
    $fgPid = 0; [void][Win32Fg]::GetWindowThreadProcessId($h, [ref]$fgPid)
    if ($fgPid -eq $ProcessId) { [System.Windows.Forms.SendKeys]::SendWait("{ESC}{ESC}"); Start-Sleep -Milliseconds 400 }
}
function Escape-SendKeys([string]$s) { [regex]::Replace($s, '([+^%~(){}\[\]])', '{$1}') }

# 1. Ventana de scripting
$main = $null
for ($i = 0; $i -lt 40 -and -not $main; $i++) { $main = Get-MainWindow; if (-not $main) { Start-Sleep -Milliseconds 500 } }
if (-not $main) { throw "mGBA (pid $ProcessId) no tiene ventana" }
if (-not (Get-ScriptWindow)) {
    Expand-Menu $main $tools
    Invoke-MenuItem $main $scripting
    for ($i = 0; $i -lt 20 -and -not (Get-ScriptWindow); $i++) { Start-Sleep -Milliseconds 300 }
}
$sw = Get-ScriptWindow
if (-not $sw) { throw "no se abrio la ventana de scripting" }

# 2. File > Load script... en un proceso aparte (el dialogo modal cuelga UI Automation)
$child = Join-Path $env:TEMP "mgba_load_script_$ProcessId.ps1"
$loadNames = ($loadScript | ForEach-Object { "'$_'" }) -join ","
$fileNames = ($fileMenu | ForEach-Object { "'$_'" }) -join ","
@"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
`$AE = [System.Windows.Automation.AutomationElement]; `$TS = [System.Windows.Automation.TreeScope]
`$c = New-Object System.Windows.Automation.PropertyCondition(`$AE::ProcessIdProperty, $ProcessId)
`$sw = @(`$AE::RootElement.FindAll(`$TS::Children, `$c)) | Where-Object { `$_.Current.Name -eq 'Scripting' } | Select-Object -First 1
function Item(`$names) { foreach (`$n in `$names) { `$f = `$sw.FindFirst(`$TS::Descendants, (New-Object System.Windows.Automation.PropertyCondition(`$AE::NameProperty, `$n))); if (`$f) { return `$f } } }
(Item @($fileNames)).GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand()
Start-Sleep -Milliseconds 500
(Item @($loadNames)).GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
"@ | Set-Content $child -Encoding ASCII
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $child -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt 40 -and -not $ready; $i++) { Start-Sleep -Milliseconds 300; $ready = Foreground-IsOurDialog }
if (-not $ready) { throw "no aparecio el dialogo de carga del script" }
# La ruta se pone con WM_SETTEXT en el campo de nombre y se pulsa el boton Abrir (BM_CLICK): SendKeys perdia la primera
# letra ("C:\..." llegaba como ":\...") y Windows respondia "El nombre de archivo no es valido".
Start-Sleep -Milliseconds 800  # el dialogo acaba de aparecer: dejarlo terminar de crearse
$dlg = [Win32Fg]::GetForegroundWindow()
$combo = [Win32Fg]::FindWindowEx($dlg, [IntPtr]::Zero, "ComboBoxEx32", $null)
$edit = [IntPtr]::Zero
if ($combo -ne [IntPtr]::Zero) {
    $inner = [Win32Fg]::FindWindowEx($combo, [IntPtr]::Zero, "ComboBox", $null)
    if ($inner -ne [IntPtr]::Zero) { $edit = [Win32Fg]::FindWindowEx($inner, [IntPtr]::Zero, "Edit", $null) }
}
$okButton = [Win32Fg]::GetDlgItem($dlg, 1)
if ($edit -eq [IntPtr]::Zero -or $okButton -eq [IntPtr]::Zero) { throw "no encuentro el campo de nombre o el boton Abrir del dialogo" }
# Hasta que el dialogo se cierre cualquier llamada de UI Automation se cuelga; el clic se reintenta si no lo atendio.
for ($attempt = 0; $attempt -lt 4 -and [Win32Fg]::IsWindow($dlg); $attempt++) {
    [void][Win32Fg]::SendText($edit, 0x000C, [IntPtr]::Zero, $ScriptPath)  # WM_SETTEXT
    Start-Sleep -Milliseconds 400
    [void][Win32Fg]::SendMsg($okButton, 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero)  # BM_CLICK
    for ($i = 0; $i -lt 10 -and [Win32Fg]::IsWindow($dlg); $i++) { Start-Sleep -Milliseconds 300 }
}
if ([Win32Fg]::IsWindow($dlg)) { throw "el dialogo de carga sigue abierto (ruta no valida?)" }
Start-Sleep -Seconds 2
Remove-Item $child -ErrorAction SilentlyContinue

# 3. Avance rapido sin limite
# Tras cerrarse el dialogo el arbol de UI Automation de la ventana tarda en refrescarse: se reintenta con la ventana
# vuelta a buscar cada vez.
function Retry-Main([scriptblock]$action, [string]$what) {
    $last = ''
    for ($i = 0; $i -lt 20; $i++) {
        try { $w = Get-MainWindow; if ($w) { return (& $action $w) } else { $last = 'sin ventana principal' } } catch { $last = $_.Exception.Message }
        Start-Sleep -Milliseconds 500
    }
    throw "no pude: $what ($last)"
}
Start-Sleep -Seconds 1
Retry-Main { param($w) Expand-Menu $w $emulation } "abrir el menu Emulacion"
Retry-Main { param($w) Expand-Menu $w $fastSpeed } "abrir Velocidad de avance rapido"
Retry-Main { param($w) Invoke-MenuItem $w $unbounded } "elegir Sin limite"
Retry-Main { param($w) Expand-Menu $w $emulation } "volver a abrir el menu Emulacion"
$main = Get-MainWindow
$ff = Find-Item $main $fastForward
if (-not $ff) { throw "no encuentro 'Avance rapido'" }
$state = $ff.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Current.ToggleState
if ($state -ne "On") { $ff.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke() }
Start-Sleep -Milliseconds 500
Close-Menus
"OK"
