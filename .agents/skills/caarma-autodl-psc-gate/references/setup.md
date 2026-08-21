# One-time setup

## AutoDL API

Create `~/.codex/caarma-cloud.env`:

```bash
AUTODL_TOKEN=developer-token
AUTODL_INSTANCE_UUID=pro-xxxxxxxxxxxx
AUTODL_SSH_KEY=/Users/USERNAME/.ssh/id_ed25519_autodl
AUTODL_SSH_USER=root
```

Protect it:

```bash
chmod 600 ~/.codex/caarma-cloud.env
```

The official API is `https://api.autodl.com/api/v1/dev/instance/pro/`.
The token is available in AutoDL Console → Account → Settings → Developer Token.

## AutoDL SSH key

The instance snapshot supplies a proxy host and SSH port. Add the selected Mac
public key once to `/root/.ssh/authorized_keys` inside the AutoDL container.
The data/environment persist after power-off. Verify key login before automation.

## Required branch contract

Every new CAARMA branch should provide:

- a CLI `--config` training entry;
- an AutoDL gate config using existing AutoDL dataset paths;
- a PSC config using `CAARMA_*` environment paths;
- a PSC Slurm script below `scripts/psc/`;
- independent output directories.

AutoDL gate configs should run two epochs so the monitor can observe Epoch 0
validation and entry into Epoch 1. The gate stops immediately after PASS.

Official AutoDL API reference:
https://www.autodl.com/docs/instance_pro_api/

