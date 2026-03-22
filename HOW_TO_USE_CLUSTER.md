# Guide

### Pre-requisites:

1. Apply for SoC account: https://mysoc.nus.edu.sg/~newacct/
2. Enable Cluster access using SoC account: https://mysoc.nus.edu.sg/~myacct/services.cgi
3. FortiClient VPN - download and set-up: https://dochub.comp.nus.edu.sg/cf/guides/network/vpn/start
4. Create SSH Key-pair to enable easier login, run the following in Windows/Terminal:
    ```bash
    ssh-keygen -t ed25519 -C "your_nusnet_email@nus.edu.sg"
    ```
5. To configure your SSH client, add your SoC cluster config to `~/.ssh/config`:
    - Note: Replace `<your_soc_username>` with your SoC account name.

    ```sshconfig
    Host soc-cluster
        HostName xlogin.comp.nus.edu.sg
        User <your_soc_username>
        IdentityFile ~/.ssh/id_ed25519
        ForwardAgent yes
        ForwardX11 no
    ```
6. (Optional) Copy Your Public Key to the cluster:

    ```bash
    ssh-copy-id soc-cluster
    ```

    If you get errors, paste the contents of `~/.ssh/id_ed25519.pub` into your account's `~/.ssh/authorized_keys` file manually on the cluster.

### Access SoC Compute Cluster

1. Turn on SoC VPN from the installed FortiClient VPN application.
2. In terminal, SSH to `soc-cluster` to access the Login Node:
    ```bash
    ssh soc-cluster
    ```
    or
    ```bash
    ssh <your_soc_username>@xlogin.comp.nus.edu.sg
    ```
3. Request for a Compute node and open interactive session in the compute node (DO NOT run heavy processes on the login nodes! Admin will kick you!):
    ```bash
    salloc --nodes=1 --ntasks=1 --time=4:00:00 --exclude=xcng[0-1]
    srun --pty bash
    ```
    - Note: Modify the time-out parameter according to own needs.
4. Install VScode & Miniconda (First-time only):
    ```bash
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz

    tar -xf vscode_cli.tar.gz
    ```
    
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
    ```
    - Refer to Miniconda Installation Documentation for more details: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer

5. Start the tunnel for VSCode:

    ```bash
    ./code tunnel
    ```
    - Note: For the first-time login, there will be some authentication required either via own Github/Microsoft account, follow instructions (recommend to use GitHub login)
6. In another VSCode window or in your browser, connect to Tunnel:
    - Open the Command Palette: Press Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS).
    - Run the "Connect to Tunnel" command: Type "Remote Tunnels: Connect to Tunnel..." in the Command Palette and select the corresponding command.

#### SLURM batch job file configuration (e.g. train.sh):

1. Modify the email address such that the compute start and end times will be sent your email account:
```bash
#SBATCH --mail-user=user@comp.nus.edu.sg
```

2. Modify the request gpu configuration to be used for model training/inference:

Example: Command to request 1 H100 GPU with 47GB memory. 

```bash
#SBATCH --partition=gpu --gres=gpu:h100-47:1
```

- See Soc Cluster page for types of available GPUs: https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware (connect to soc vpn to access this site)


#### To submit SLURM batch jobs via the Login node:

```bash
sbatch <your_slurm_job_filename.sh>
```

#### Monitor your job after submitting batch job

```bash
squeue -u <your_soc_username>
```