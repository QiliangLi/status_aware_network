' Hidden-window wrapper: runs tools/auto_pull.sh via Git Bash without flashing a console window.
' Called by Windows scheduled task StatusAwareNetwork_AutoPull every minute (win machine only).
' NOTE: keep this file ASCII-only. VBScript reads non-BOM files as ANSI; Chinese comments break it.
' bash.exe path is machine-specific (Git installed at C:\Users\liqiliang\Apps\Git).
CreateObject("WScript.Shell").Run _
  """C:\Users\liqiliang\Apps\Git\usr\bin\bash.exe"" -lc ""/d/Coding/status_aware_network/tools/auto_pull.sh""", 0, False
