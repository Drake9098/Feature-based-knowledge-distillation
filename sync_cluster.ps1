param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("upload", "download", "download-logs", "download-checkpoints", "download-wandb", "sync-wandb", "push", "pull", "show-ssh-pubkey", "verify-ssh")]
    [string]$Action,

    [Parameter(Mandatory = $false)]
    [string]$Path,  # per push/pull: path relativo al repo, es. "src/training/train_baseline.py" o "cluster/"

    # Se impostato (es. "gcluster"), ssh/scp usano solo questo Host come in ~/.ssh/config (vedi SSH_PASSWORDLESS_UPLOAD_GUIDE.md §4).
    [Parameter(Mandatory = $false)]
    [string]$SshConfigHost = ""
)

$CLUSTER_USER = "rtisvt98h23c351g"
$CLUSTER_HOST = "gcluster.dmi.unict.it"
# Nome cartella sul cluster (una sola variabile per tutti i comandi ssh/scp)
$REMOTE_DIR = "Feature-based-knowledge-distillation"
$LOCAL = $PSScriptRoot

# --- SSH: allineato a SSH_PASSWORDLESS_UPLOAD_GUIDE.md ---
# - Chiave: ordine 1) $env:SYNC_CLUSTER_SSH_IDENTITY  2) ~/.ssh/id_ed25519  3) ~/.ssh/id_ed25519_gcluster_dmi
# - Registra sul server il .pub corrispondente in ~/.ssh/authorized_keys; opzionale: ssh-agent + ssh-add (guida §2)
# - Consigliato in %USERPROFILE%\.ssh\config:
#     Host gcluster
#       HostName gcluster.dmi.unict.it
#       User <codice-fiscale>
#       IdentityFile ~/.ssh/id_ed25519
#       IdentitiesOnly yes
#   Poi: .\sync_cluster.ps1 ... -SshConfigHost gcluster
if ($SshConfigHost) {
    $SSH_TARGET = $SshConfigHost
    $REMOTE = "${SshConfigHost}:~/${REMOTE_DIR}"
} else {
    $SSH_TARGET = "${CLUSTER_USER}@${CLUSTER_HOST}"
    $REMOTE = "${CLUSTER_USER}@${CLUSTER_HOST}:~/${REMOTE_DIR}"
}

function Get-SshIdentityPath {
    if ($env:SYNC_CLUSTER_SSH_IDENTITY -and (Test-Path -LiteralPath $env:SYNC_CLUSTER_SSH_IDENTITY)) {
        return $env:SYNC_CLUSTER_SSH_IDENTITY
    }
    $candidates = @(
        (Join-Path $env:USERPROFILE ".ssh\id_ed25519"),
        (Join-Path $env:USERPROFILE ".ssh\id_ed25519_gcluster_dmi")
    )
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Get-SshIdentityArgs {
    $path = Get-SshIdentityPath
    if ($path) {
        # IdentitiesOnly=yes evita che il client provi altre chiavi prima di quella indicata (guida §4).
        return @("-i", $path, "-o", "IdentitiesOnly=yes")
    }
    return @()
}

# Esclusioni tar (symlink o run “latest” che duplicano dati)
$TAR_EXCLUDES = @("--exclude=latest", "--exclude=latest-run")

function Download-RemoteDir($remoteSubpath, $localDest) {
    $idArgs = Get-SshIdentityArgs
    New-Item -ItemType Directory -Force -Path $localDest | Out-Null
    $excludeArgs = $TAR_EXCLUDES -join " "
    ssh @idArgs $SSH_TARGET "cd ~/${REMOTE_DIR} && tar cf - $excludeArgs $remoteSubpath" | tar xvf - -C "$LOCAL"
}

function Upload {
    Write-Host "Uploading project to cluster..." -ForegroundColor Cyan
    $idArgs = Get-SshIdentityArgs
    $idPath = Get-SshIdentityPath
    if ($idArgs.Count -gt 0) {
        Write-Host "  SSH target:   $SSH_TARGET" -ForegroundColor Gray
        Write-Host "  SSH identity: $idPath" -ForegroundColor Gray
    } else {
        Write-Host "  Nessuna chiave in id_ed25519 / id_ed25519_gcluster_dmi / `$env:SYNC_CLUSTER_SSH_IDENTITY — uso default client (ssh-agent)." -ForegroundColor DarkYellow
        Write-Host "  Vedi SSH_PASSWORDLESS_UPLOAD_GUIDE.md §1–2." -ForegroundColor DarkYellow
    }

    Write-Progress -Activity "Upload" -Status "Cleaning __pycache__..." -PercentComplete 0
    Get-ChildItem -Path $LOCAL -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Write-Progress -Activity "Upload" -Status "Creating remote directories..." -PercentComplete 2
    ssh @idArgs $SSH_TARGET "mkdir -p ~/${REMOTE_DIR}/configs ~/${REMOTE_DIR}/experiments/logs ~/${REMOTE_DIR}/experiments/checkpoints ~/${REMOTE_DIR}/dataset ~/${REMOTE_DIR}/weights"

    # dataset/ e weights/: upload incrementale (nessun rm -rf remoto); se assenti in locale → [SKIP].
    $dirsNoFullReplace = @("dataset", "weights")
    $items = @(
        "src",
        "cluster",
        "configs",
        "scripts",
        "dataset",
        "weights",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "action_plan.md"
    )
    if (Test-Path (Join-Path $LOCAL ".env")) {
        $items += ".env"
    }

    # Upload per cartella con scp -r (come sync_cluster_dani.ps1): poche connessioni SSH.
    # Il vecchio upload file-per-file causava una richiesta password / handshake per ogni file
    # se la chiave non era usata (es. chiave non registrata o path -i sbagliato).
    $dirItems = [System.Collections.Generic.List[string]]::new()
    $fileItems = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $items) {
        $localPath = Join-Path $LOCAL $item
        if (-not (Test-Path $localPath)) {
            Write-Host "  [SKIP] $item (not found)" -ForegroundColor Yellow
            continue
        }
        if (Test-Path $localPath -PathType Container) {
            $dirItems.Add($item)
        } else {
            $fileItems.Add($item)
        }
    }

    $dirsToWipe = $dirItems | Where-Object { $_ -notin $dirsNoFullReplace }
    if ($dirsToWipe.Count -gt 0) {
        $rmCmd = ($dirsToWipe | ForEach-Object { "rm -rf ~/${REMOTE_DIR}/$_" }) -join "; "
        ssh @idArgs $SSH_TARGET $rmCmd
    }

    $step = 0
    $totalSteps = $dirItems.Count + $fileItems.Count
    foreach ($item in $dirItems) {
        $step++
        $localPath = Join-Path $LOCAL $item
        $pct = [int](($step / [math]::Max($totalSteps, 1)) * 100)

        # --- NUOVO CONTROLLO: Salta scp se è una directory protetta ed esiste già ---
        if ($item -in $dirsNoFullReplace) {
            # Controlla sul server se la cartella esiste e non è vuota
            $remoteCheck = ssh @idArgs $SSH_TARGET "if [ -d ~/${REMOTE_DIR}/$item ] && [ ""`$(ls -A ~/${REMOTE_DIR}/$item 2>/dev/null)"" ]; then echo 'EXISTS'; fi"
            if ($remoteCheck -match 'EXISTS') {
                Write-Progress -Activity "Upload" -Status "[$step/$totalSteps] SKIP $item/" -PercentComplete $pct
                Write-Host "  [SKIP] $item/ (già presente sul cluster)" -ForegroundColor Yellow
                continue # Salta direttamente il trasferimento
            }
            $mode = "(nuovo upload)"
        } else {
            $mode = ""
        }
        # ----------------------------------------------------------------------------

        Write-Progress -Activity "Upload" -Status "[$step/$totalSteps] scp -r $item/ $mode" -PercentComplete $pct
        Write-Host "  Copiando $item/ $mode ..." -ForegroundColor Gray
        # Se la cartella remota non esiste (o l'abbiamo appena creata), scp -r copierà l'intera directory
        scp @idArgs -r -q "$localPath" "${REMOTE}/"
    }

    foreach ($item in $fileItems) {
        $step++
        $localPath = Join-Path $LOCAL $item
        $pct = [int](($step / [math]::Max($totalSteps, 1)) * 100)
        Write-Progress -Activity "Upload" -Status "[$step/$totalSteps] $item" -PercentComplete $pct
        Write-Host "  Copiando $item ..." -ForegroundColor Gray
        scp @idArgs -q "$localPath" "${REMOTE}/"
    }

    Write-Progress -Activity "Upload" -Completed
    Write-Host "Upload complete ($totalSteps trasferimenti: $($dirItems.Count) cartelle + $($fileItems.Count) file)." -ForegroundColor Green
    Write-Host "Sul cluster: bash cluster/check_offline_assets.sh (verifica CIFAR-100 + pesi teacher)." -ForegroundColor Gray
}

function DownloadAll {
    Write-Host "Downloading all outputs from cluster..." -ForegroundColor Cyan

    Write-Progress -Activity "Download" -Status "[1/2] logs..." -PercentComplete 0
    DownloadLogs

    Write-Progress -Activity "Download" -Status "[2/2] checkpoints..." -PercentComplete 50
    DownloadCheckpoints

    Write-Progress -Activity "Download" -Completed
    Write-Host "Download complete." -ForegroundColor Green
}

function DownloadLogs {
    Write-Progress -Activity "Download" -Status "Downloading logs/..." -PercentComplete 0
    $dest = Join-Path $LOCAL "experiments\\logs"
    Download-RemoteDir "experiments/logs" $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to experiments\\logs\\" -ForegroundColor Gray
}

function DownloadCheckpoints {
    Write-Progress -Activity "Download" -Status "Downloading checkpoints/..." -PercentComplete 0
    $dest = Join-Path $LOCAL "experiments\\checkpoints"
    Download-RemoteDir "experiments/checkpoints" $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to experiments\\checkpoints\\" -ForegroundColor Gray
}

function DownloadWandb {
    Write-Progress -Activity "Download" -Status "Downloading experiments/logs/ (cerca wandb offline sotto experiments/logs/)..." -PercentComplete 0

    $dest = Join-Path $LOCAL "experiments\\logs"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Download-RemoteDir "experiments/logs" $dest

    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved under experiments\\logs\\" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Per sincronizzare run offline su wandb.ai:" -ForegroundColor Yellow
    Write-Host "  .\sync_cluster.ps1 -Action sync-wandb" -ForegroundColor Yellow
}

function SyncWandb {
    Write-Host "Syncing wandb offline runs to wandb.ai..." -ForegroundColor Cyan

    $venvActivate = Join-Path $LOCAL ".venv\Scripts\Activate.ps1"
    if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) {
        if (Test-Path $venvActivate) {
            Write-Host "  Activating .venv for wandb CLI..." -ForegroundColor Gray
            & $venvActivate
        }
        if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) {
            Write-Host "wandb CLI not found. Install: pip install wandb" -ForegroundColor Red
            return
        }
    }

    $logsDir = Join-Path $LOCAL "experiments\\logs"
    if (-not (Test-Path $logsDir)) {
        Write-Host "No experiments\\logs\\ found. Run download-wandb first." -ForegroundColor Red
        return
    }

    $wandbDirs = Get-ChildItem -Path $logsDir -Recurse -Directory -Filter "wandb" |
        Where-Object { (Get-ChildItem -Path $_.FullName -Directory -Filter "offline-run-*").Count -gt 0 }

    if ($wandbDirs.Count -eq 0) {
        Write-Host "No offline runs found under logs\" -ForegroundColor Yellow
        return
    }

    $totalRuns = 0
    foreach ($wdir in $wandbDirs) {
        $totalRuns += (Get-ChildItem -Path $wdir.FullName -Directory -Filter "offline-run-*").Count
    }
    Write-Host "Found $totalRuns offline run(s) in $($wandbDirs.Count) wandb dir(s):" -ForegroundColor Gray
    $synced = 0
    $failed = 0
    foreach ($wdir in $wandbDirs) {
        $offlineRuns = Get-ChildItem -Path $wdir.FullName -Directory -Filter "offline-run-*"
        foreach ($run in $offlineRuns) {
            Write-Host "  [$($synced + $failed + 1)/$totalRuns] Syncing $($run.Name) ..." -ForegroundColor Gray -NoNewline
            $result = & wandb sync --include-synced $run.FullName 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host " OK" -ForegroundColor Green
                $synced++
            } else {
                Write-Host " FAILED" -ForegroundColor Red
                Write-Host ($result | Out-String) -ForegroundColor DarkRed
                $failed++
            }
        }
    }

    Write-Host ""
    Write-Host "Sync complete: $synced succeeded, $failed failed." -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
}

function Push {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action push -Path <file-or-folder>" -ForegroundColor Red
        return
    }
    $idArgs = Get-SshIdentityArgs
    $localPath = Join-Path $LOCAL $Path
    if (-not (Test-Path $localPath)) {
        Write-Host "Not found: $Path" -ForegroundColor Red
        return
    }
    $remotePath = $Path -replace '\\', '/'
    $remoteDir = ($remotePath | Split-Path) -replace '\\', '/'
    if ($remoteDir) {
        ssh @idArgs $SSH_TARGET "mkdir -p ~/${REMOTE_DIR}/$remoteDir"
    }
    if (Test-Path $localPath -PathType Container) {
        ssh @idArgs $SSH_TARGET "mkdir -p ~/${REMOTE_DIR}/$remotePath"
        scp @idArgs -rq "$localPath/." "${REMOTE}/$remotePath/"
    } else {
        scp @idArgs -q $localPath "${REMOTE}/$remotePath"
    }
    Write-Host "Pushed $Path -> cluster (~/${REMOTE_DIR})" -ForegroundColor Green
}

function Pull {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action pull -Path <file-or-folder>" -ForegroundColor Red
        return
    }
    $idArgs = Get-SshIdentityArgs
    $remotePath = $Path -replace '\\', '/'
    $localPath = Join-Path $LOCAL $Path
    $localDir = Split-Path $localPath
    if ($localDir) {
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    }
    scp @idArgs -rq "${REMOTE}/$remotePath" $localPath
    Write-Host "Pulled $Path <- cluster (~/${REMOTE_DIR})" -ForegroundColor Green
}

function Show-SshPubKey {
    $priv = Get-SshIdentityPath
    if (-not $priv) {
        Write-Host "Nessuna chiave privata trovata (id_ed25519, id_ed25519_gcluster_dmi o SYNC_CLUSTER_SSH_IDENTITY)." -ForegroundColor Red
        Write-Host "Genera una coppia (guida SSH_PASSWORDLESS_UPLOAD_GUIDE.md §1), es.:" -ForegroundColor Yellow
        Write-Host "  ssh-keygen -t ed25519 -C `"gcluster`"" -ForegroundColor Gray
        return
    }
    $pub = "${priv}.pub"
    if (-not (Test-Path -LiteralPath $pub)) {
        Write-Host "Manca il file pubblico: $pub" -ForegroundColor Red
        return
    }
    Write-Host "Chiave usata dallo script: $priv" -ForegroundColor Gray
    Write-Host "Incolla la riga seguente in ~/.ssh/authorized_keys sul cluster (una riga, permessi 600):" -ForegroundColor Cyan
    Get-Content -LiteralPath $pub
}

function Test-ClusterSsh {
    Write-Host "Verifica SSH (echo SSH_OK sul cluster)..." -ForegroundColor Cyan
    $idArgs = Get-SshIdentityArgs
    $idPath = Get-SshIdentityPath
    if ($idPath) {
        Write-Host "  Identity: $idPath" -ForegroundColor Gray
    }
    Write-Host "  Target:   $SSH_TARGET" -ForegroundColor Gray
    ssh @idArgs $SSH_TARGET "echo SSH_OK"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK — connessione riuscita." -ForegroundColor Green
    } else {
        Write-Host "Fallita (exit $LASTEXITCODE). Guida: SSH_PASSWORDLESS_UPLOAD_GUIDE.md" -ForegroundColor Red
    }
}

switch ($Action) {
    "upload"                { Upload }
    "download"              { DownloadAll }
    "download-logs"         { DownloadLogs }
    "download-checkpoints"  { DownloadCheckpoints }
    "download-wandb"        { DownloadWandb }
    "sync-wandb"            { SyncWandb }
    "push"                  { Push }
    "pull"                  { Pull }
    "show-ssh-pubkey"       { Show-SshPubKey }
    "verify-ssh"            { Test-ClusterSsh }
}
