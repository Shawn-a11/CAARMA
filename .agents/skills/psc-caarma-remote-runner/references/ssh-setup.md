# SSH setup on macOS

PSC documents `bridges2.psc.edu` on port 22 as the login endpoint. Public keys
must be registered with PSC before key authentication works.

Recommended `~/.ssh/config` entry:

```sshconfig
Host bridges2
  HostName bridges2.psc.edu
  User sge2
  ControlMaster auto
  ControlPath ~/.ssh/cm-%C
  ControlPersist 8h
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

For unattended Codex calls, register an SSH public key with PSC and verify:

```bash
ssh -o BatchMode=yes bridges2 'hostname; id -un; command -v sbatch'
```

If key registration is not ready, establish one interactive master connection:

```bash
ssh bridges2
```

Authenticate manually, then exit. `ControlPersist` lets subsequent Codex SSH
commands reuse the authenticated socket. The skill never stores a PSC password.

Official guide: https://www.psc.edu/resources/bridges-2/user-guide/

