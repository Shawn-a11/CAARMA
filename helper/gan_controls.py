"""Pure helpers for paper-aligned GAN ablations."""


VALID_GAN_SCHEDULES = {"paired", "source_state_machine"}
VALID_LAMBDA_MODES = {"dynamic", "fixed"}


def resolve_gan_config(config, main_lr, weight_decay):
    """Validate and return the GAN controls that training will actually use."""
    resolved = {
        "main_lr": float(main_lr),
        "discriminator_lr": float(config.get("discriminator_lr", 0.0002)),
        "weight_decay": float(weight_decay),
        "lambda_adv_mode": str(config.get("lambda_adv_mode", "dynamic")),
        "lambda_adv_fixed": float(config.get("lambda_adv_fixed", 0.05)),
        "lambda_adv_init": float(config.get("lambda_adv_init", 0.25)),
        "lambda_adv_floor": float(config.get("lambda_adv_floor", 0.0001)),
        "lambda_adv_cap": float(config.get("lambda_adv_cap", 0.01)),
        "lambda_adv_pretrain": float(config.get("lambda_adv_pretrain", 0.0005)),
        "gan_schedule": str(config.get("gan_schedule", "paired")),
        "pretrain_epochs": int(config.get("pretrain_epochs", 15)),
        "pretrain_g_steps": int(config.get("pretrain_g_steps", 5)),
    }

    if resolved["main_lr"] <= 0:
        raise ValueError("init_lr must be positive")
    if resolved["discriminator_lr"] <= 0:
        raise ValueError("discriminator_lr must be positive")
    if resolved["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if resolved["lambda_adv_mode"] not in VALID_LAMBDA_MODES:
        raise ValueError(
            f"Unsupported lambda_adv_mode: {resolved['lambda_adv_mode']}"
        )
    if resolved["gan_schedule"] not in VALID_GAN_SCHEDULES:
        raise ValueError(f"Unsupported gan_schedule: {resolved['gan_schedule']}")
    if resolved["pretrain_epochs"] < 0:
        raise ValueError("pretrain_epochs must be non-negative")
    if resolved["pretrain_g_steps"] < 1:
        raise ValueError("pretrain_g_steps must be at least 1")

    lambda_keys = (
        "lambda_adv_fixed",
        "lambda_adv_init",
        "lambda_adv_floor",
        "lambda_adv_cap",
        "lambda_adv_pretrain",
    )
    if any(resolved[key] < 0 for key in lambda_keys):
        raise ValueError("lambda_adv values must be non-negative")
    if resolved["lambda_adv_floor"] > resolved["lambda_adv_cap"]:
        raise ValueError("lambda_adv_floor cannot exceed lambda_adv_cap")

    return resolved


def scheduled_updates(
    schedule,
    epoch,
    batch_idx,
    pretrain_epochs=15,
    pretrain_g_steps=5,
):
    """Return the optimizer updates to execute for one training batch."""
    if schedule not in VALID_GAN_SCHEDULES:
        raise ValueError(f"Unsupported gan_schedule: {schedule}")
    if pretrain_g_steps < 1:
        raise ValueError("pretrain_g_steps must be at least 1")

    if schedule == "paired":
        return ("d", "g")

    if epoch <= pretrain_epochs:
        cycle_position = batch_idx % (pretrain_g_steps + 1)
        return ("g",) if cycle_position < pretrain_g_steps else ("d",)

    return ("d",) if batch_idx % 2 == 0 else ("g",)


def generator_lambda_plan(
    mode,
    fixed_value,
    dynamic_value,
    pretrain_value,
    use_pretrain_value,
):
    """Return ``(lambda_value, adjust_dynamically)`` for a G update."""
    if mode not in VALID_LAMBDA_MODES:
        raise ValueError(f"Unsupported lambda_adv_mode: {mode}")
    if mode == "fixed":
        return float(fixed_value), False
    if use_pretrain_value:
        return float(pretrain_value), False
    return float(dynamic_value), True
