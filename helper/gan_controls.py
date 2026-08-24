"""Pure helpers for paper-aligned GAN ablations."""


VALID_GAN_SCHEDULES = {"paired", "source_state_machine"}
VALID_LAMBDA_MODES = {"dynamic", "fixed"}


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
