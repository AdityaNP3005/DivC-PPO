"""Deterministic measurements and artifacts; formulas retained from the original validator."""


def run(context):
    # Imported here, not at module top: validate_inference imports this module before
    # setup() launches Isaac, and rl_env.env needs the running app.
    import math
    import os
    import sys

    import numpy as np
    import torch

    from evaluation.validation_contract import (
        atomic_json, completion_status, require_unchanged_assets, require_valid_metrics,
    )
    from rl_env.env import (
        EE_LINK_NAME,
        GT_SUCCESS_XY_TOL_M,
        GT_SUCCESS_Z_TOL_M,
        JOINT_NAMES_EXPR,
        SCRIPTED_STAY_STEPS_REQUIRED,
    )
    from rl_env.env_ids import find_body_id
    from rl_env.rewards.error_sources import gt_target_error_w
    from rl_env.success_metrics import advance_stable_success, per_axis_in_band

    _validation_context = context['_validation_context']
    args_cli = context['args_cli']
    checkpoint_path = context['checkpoint_path']
    checkpoint_tag = context['checkpoint_tag']
    env = context['env']
    env_cfg = context['env_cfg']
    isaac_env = context['isaac_env']
    model = context['model']
    out_dir = context['out_dir']
    simulation_app = context['simulation_app']

    # ─── 6. EE body id ───────────────────────────────────────────────────────────
    robot = isaac_env.scene["robot"]
    ee_id = find_body_id(robot, EE_LINK_NAME)
    if ee_id is None:
        print(f"[ERROR] EE body '{EE_LINK_NAME}' not found.", flush=True)
        env.close(); simulation_app.close(); sys.exit(1)
    NUM_ENVS = int(args_cli.num_envs)
    DEV = robot.data.body_pos_w.device

    # Arm joint ids (6 DOF driven by the IK) — for the hold-phase activity metrics below.
    _arm_joint_ids, _arm_joint_names = robot.find_joints(JOINT_NAMES_EXPR, preserve_order=True)
    ARM_JIDS = torch.as_tensor(list(_arm_joint_ids), dtype=torch.long, device=DEV)
    print(f"[VALIDATE] arm joints for motor metrics: {list(_arm_joint_names)}", flush=True)

    _NEAR_FACTOR = 2.0


    def _compute_gt():
        """Mirror monitor_stats._accumulate_gt_stats GT computation (per step).

        Returns (gt_error_w, gt_at, near_now, ee_pos_w) — ee_pos_w is also used by the
        jerk metric (3rd time-derivative of the EE Cartesian world position).
        """
        ee_pos_w = robot.data.body_pos_w[:, ee_id]
        gt_error_w = gt_target_error_w(isaac_env, ee_id)
        gt_at = per_axis_in_band(gt_error_w, GT_SUCCESS_XY_TOL_M, GT_SUCCESS_Z_TOL_M)
        near_now = per_axis_in_band(
            gt_error_w,
            float(GT_SUCCESS_XY_TOL_M) * _NEAR_FACTOR,
            float(GT_SUCCESS_Z_TOL_M) * _NEAR_FACTOR,
        )
        return gt_error_w.detach(), gt_at.detach(), near_now.detach(), ee_pos_w.detach()


    # ─── 7. Loop validasi ────────────────────────────────────────────────────────
    if args_cli.max_steps and int(args_cli.max_steps) > 0:
        MAX_STEPS = int(args_cli.max_steps)
    else:
        waves = math.ceil(args_cli.min_episodes / max(NUM_ENVS, 1)) + 2
        MAX_STEPS = waves * 280   # ~280 > horizon episode (~210) → margin aman
    MIN_EP = int(args_cli.min_episodes)

    # Control-step period (s): EE jerk = 3rd time-derivative of position over control steps.
    # ManagerBasedRLEnv.step_dt = decimation × sim.dt (=2/60 → 30 Hz here); fall back to that.
    try:
        DT = float(isaac_env.step_dt)
    except Exception:
        DT = float(getattr(env_cfg, "decimation", 2)) * float(env_cfg.sim.dt)
    DT3 = DT ** 3
    STAY_REQ = int(SCRIPTED_STAY_STEPS_REQUIRED)   # 20 consecutive in-band steps (grasp trigger)
    print(f"[VALIDATE] control dt={DT:.5f}s | stay_req={STAY_REQ} steps | success tol xy="
          f"{GT_SUCCESS_XY_TOL_M*1000:.1f}mm z={GT_SUCCESS_Z_TOL_M*1000:.1f}mm", flush=True)

    # Per-episode records (final step of each env that just reset)
    rec_exy, rec_ez, rec_norm = [], [], []
    rec_ex, rec_ey, rec_ez_s = [], [], []   # signed → bias/jitter
    rec_succ, rec_reach, rec_hold, rec_near = [], [], [], []
    rec_jerk, rec_jerk_peak = [], []        # per-episode RMS + peak EE jerk (m/s³)
    rec_steps_stay20 = []                   # per-episode control steps to first reach STAY_REQ consecutive
    # ── Action-fluctuation records (per episode) ──
    # Deterministic-inference measures of how much the COMMAND still moves. Split by PHASE
    # because the arm spends ~85% of an episode parked in-band: during hold there is no net
    # motion, so every command delta is fluctuation with no useful work done. These are
    # simulated kinematic quantities — no torque or current is measured, so do NOT report
    # them as actuator load or wear.
    rec_dact_hold, rec_dact_transit = [], []   # RMS |Δa| (action units) per phase
    rec_dact_rev_hold = []                     # Δa sign flips per axis per step, during hold
    rec_jvel_hold, rec_jvel_transit = [], []   # RMS joint velocity (rad/s) per phase
    rec_jvel_rev_hold = []                     # joint-velocity sign flips per joint per step, during hold

    obs = env.reset()
    gt_error_w, gt_at, near_now, ee_pos = _compute_gt()
    ever_at = torch.zeros_like(gt_at)
    hold_count = torch.zeros_like(gt_at, dtype=torch.long)
    ever_near = torch.zeros_like(near_now)
    last_error = gt_error_w.clone()
    last_ep_len = isaac_env.episode_length_buf.detach().clone()

    # ── Jerk state (per-env): rolling last-3 EE positions + running |jerk| accumulators ──
    _z = torch.zeros(NUM_ENVS, device=DEV)
    _zl = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEV)
    p1 = ee_pos.clone(); p2 = ee_pos.clone(); p3 = ee_pos.clone()
    jerk_hist = _zl.clone()          # #samples stored this episode (need ≥3 for a 4-point stencil)
    jerk_sqsum = _z.clone(); jerk_max = _z.clone(); jerk_cnt = _zl.clone()

    # ── Stay-20 state (per-env): consecutive in-band streak + first-reach step ──
    # env.reset() does not run the interval event, so the initial observation is not a
    # completed control step. Start at zero; the first env.step() contributes step 1.
    stay_streak = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEV)
    stay20_done = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEV)
    stay20_step = torch.full((NUM_ENVS,), float("nan"), device=DEV)

    # ── Action-fluctuation state (per-env), same reset discipline as the jerk accumulators ──
    # Δa needs 1 prior action from THIS episode; the sign-flip count needs 2 (a prior Δa).
    _ACT_DIM = int(model.action_space.shape[0])
    _NJ = int(ARM_JIDS.numel())
    prev_act = torch.zeros((NUM_ENVS, _ACT_DIM), device=DEV)
    prev_dact_sign = torch.zeros((NUM_ENVS, _ACT_DIM), device=DEV)
    prev_jvel_sign = torch.zeros((NUM_ENVS, _NJ), device=DEV)
    act_hist = _zl.clone()           # #actions seen this episode
    # per-phase running sums: [hold, transit]
    dact_sq = [_z.clone(), _z.clone()];  dact_cnt = [_zl.clone(), _zl.clone()]
    jvel_sq = [_z.clone(), _z.clone()];  jvel_cnt = [_zl.clone(), _zl.clone()]
    dact_rev = _z.clone();  jvel_rev = _z.clone()   # flips accumulated during hold only
    hold_steps = _zl.clone()                        # in-band steps seen this episode (rev denominator)

    print(f"[VALIDATE] target≥{MIN_EP} ep | num_envs={NUM_ENVS} | max_steps={MAX_STEPS}", flush=True)

    step = 0
    completed = 0
    while step < MAX_STEPS and completed < MIN_EP:
        actions, _ = model.predict(obs, deterministic=True)
        obs, _rew, _dones, _infos = env.step(actions)
        step += 1

        gt_error_w, gt_at, near_now, ee_pos = _compute_gt()
        ep_len = isaac_env.episode_length_buf.detach().clone()
        just_reset = ep_len < last_ep_len   # envs that just finished/reset

        # ── Jerk this step: 4-point stencil (x −3p₁ +3p₂ −p₃)/dt³ over EE world pos.
        # Only valid when p₁,p₂,p₃ are ≥3 prior samples FROM THIS episode (jerk_hist≥3) and
        # this step is NOT a reset (post-reset x is the new episode's first pose → skip).
        have3 = (jerk_hist >= 3) & (~just_reset)
        jvec = (ee_pos - 3.0 * p1 + 3.0 * p2 - p3) / DT3
        jmag = jvec.norm(dim=1)

        # ── Activity this step: command delta |Δa| + joint velocity, bucketed by phase.
        # Needs ≥1 prior action FROM THIS episode; act_hist is zeroed at reset so the delta
        # spanning an episode boundary (old episode's last action → new episode's first) is
        # never counted. Flips additionally need a prior Δa (act_hist ≥ 2).
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=DEV)
        have_a = (act_hist >= 1) & (~just_reset)
        dact = act_t - prev_act
        dact_mag = dact.norm(dim=1)
        jvel = robot.data.joint_vel[:, ARM_JIDS].detach()
        jvel_mag = jvel.norm(dim=1)
        dact_sign, jvel_sign = torch.sign(dact), torch.sign(jvel)
        n_flip_a = ((dact_sign * prev_dact_sign) < 0).sum(dim=1).to(_z.dtype)
        n_flip_j = ((jvel_sign * prev_jvel_sign) < 0).sum(dim=1).to(_z.dtype)

        if torch.any(just_reset):
            idx = torch.nonzero(just_reset, as_tuple=False).flatten().tolist()
            for i in idx:
                ex = float(last_error[i, 0]); ey = float(last_error[i, 1]); ez = float(last_error[i, 2])
                rec_ex.append(ex); rec_ey.append(ey); rec_ez_s.append(ez)
                rec_exy.append(math.sqrt(ex * ex + ey * ey))
                rec_ez.append(abs(ez))
                rec_norm.append(math.sqrt(ex * ex + ey * ey + ez * ez))
                reached = bool(ever_at[i])
                stable = bool(stay20_done[i])
                rec_reach.append(1.0 if reached else 0.0)
                rec_succ.append(1.0 if stable else 0.0)
                rec_near.append(1.0 if (bool(ever_near[i]) and not stable) else 0.0)
                if bool(ever_at[i]):
                    rec_hold.append(float(hold_count[i]))
                # Finished-episode jerk (RMS + peak of |jerk| over the episode's control steps).
                if int(jerk_cnt[i]) > 0:
                    rec_jerk.append(math.sqrt(float(jerk_sqsum[i]) / float(jerk_cnt[i])))
                    rec_jerk_peak.append(float(jerk_max[i]))
                # Steps to first reach STAY_REQ consecutive in-band steps (NaN = never → excluded;
                # stay20_rate is recovered as len(rec_steps_stay20)/n_ep downstream).
                if bool(stay20_done[i]):
                    rec_steps_stay20.append(float(stay20_step[i]))
                # RMS over each phase's control steps + in-band direction-flip rate
                # (normalised per step per axis → 0..1 = fraction of axes reversing each step).
                if int(dact_cnt[0][i]) > 0:
                    rec_dact_hold.append(math.sqrt(float(dact_sq[0][i]) / float(dact_cnt[0][i])))
                    rec_jvel_hold.append(math.sqrt(float(jvel_sq[0][i]) / float(jvel_cnt[0][i])))
                if int(dact_cnt[1][i]) > 0:
                    rec_dact_transit.append(math.sqrt(float(dact_sq[1][i]) / float(dact_cnt[1][i])))
                    rec_jvel_transit.append(math.sqrt(float(jvel_sq[1][i]) / float(jvel_cnt[1][i])))
                if int(hold_steps[i]) > 0:
                    rec_dact_rev_hold.append(float(dact_rev[i]) / float(hold_steps[i]) / _ACT_DIM)
                    rec_jvel_rev_hold.append(float(jvel_rev[i]) / float(hold_steps[i]) / _NJ)
            completed += len(idx)

        # Update per-env accumulators: reset envs start fresh, the others OR/increment.
        jr = just_reset
        ever_at = torch.where(jr, torch.zeros_like(gt_at), ever_at | gt_at)
        gt_at_long = gt_at.to(torch.long)
        hold_count = torch.where(jr, torch.zeros_like(gt_at_long), hold_count + gt_at_long)
        ever_near = torch.where(jr, torch.zeros_like(near_now), ever_near | near_now)

        # Jerk accumulators: add this step's |jerk| where valid, then zero the reset envs so the
        # new episode starts clean (reset envs contributed 0 via have3 anyway).
        jerk_sqsum = torch.where(jr, _z, jerk_sqsum + torch.where(have3, jmag * jmag, _z))
        jerk_cnt = torch.where(jr, _zl, jerk_cnt + have3.to(torch.long))
        jerk_max = torch.where(jr, _z, torch.where(have3 & (jmag > jerk_max), jmag, jerk_max))
        # Shift EE-position history (p₁ newest). Reset envs fill p₂/p₃ with x and restart the count.
        jr1 = jr.unsqueeze(1)
        p3 = torch.where(jr1, ee_pos, p2)
        p2 = torch.where(jr1, ee_pos, p1)
        p1 = ee_pos.clone()
        jerk_hist = torch.where(jr, torch.ones_like(_zl), jerk_hist + 1)

        # Activity accumulators: bucket this step into hold (GT in-band) or transit.
        in_hold = gt_at & have_a
        in_transit = (~gt_at) & have_a
        for k, mask in ((0, in_hold), (1, in_transit)):
            dact_sq[k] = torch.where(jr, _z, dact_sq[k] + torch.where(mask, dact_mag * dact_mag, _z))
            dact_cnt[k] = torch.where(jr, _zl, dact_cnt[k] + mask.to(torch.long))
            jvel_sq[k] = torch.where(jr, _z, jvel_sq[k] + torch.where(mask, jvel_mag * jvel_mag, _z))
            jvel_cnt[k] = torch.where(jr, _zl, jvel_cnt[k] + mask.to(torch.long))
        # Direction flips counted ONLY while parked in-band, where they do no useful work.
        flip_ok = in_hold & (act_hist >= 2)
        dact_rev = torch.where(jr, _z, dact_rev + torch.where(flip_ok, n_flip_a, _z))
        jvel_rev = torch.where(jr, _z, jvel_rev + torch.where(flip_ok, n_flip_j, _z))
        hold_steps = torch.where(jr, _zl, hold_steps + flip_ok.to(torch.long))
        prev_act = act_t
        jr1_a = jr.unsqueeze(1)
        prev_dact_sign = torch.where(jr1_a, torch.zeros_like(dact_sign), dact_sign)
        prev_jvel_sign = torch.where(jr1_a, torch.zeros_like(jvel_sign), jvel_sign)
        act_hist = torch.where(jr, torch.zeros_like(_zl), act_hist + 1)

        # Stay-20: consecutive in-band streak (resets to zero at episode reset). Record the FIRST
        # step (episode_length) at which the streak reaches STAY_REQ, once per episode.
        stay20_prev = stay20_done
        stay_streak, stay20_done = advance_stable_success(
            stay_streak, stay20_done, gt_at, jr, STAY_REQ
        )
        stay20_step = torch.where(jr, torch.full_like(stay20_step, float("nan")), stay20_step)
        reached_now = stay20_done & (~stay20_prev)
        stay20_step = torch.where(reached_now, ep_len.to(stay20_step.dtype), stay20_step)

        last_error = gt_error_w.clone()
        last_ep_len = ep_len

        if step % 50 == 0:
            print(f"[VALIDATE] step {step}/{MAX_STEPS} | episodes {completed}/{MIN_EP}", flush=True)

    print(f"[VALIDATE] done: {completed} episodes in {step} steps.", flush=True)
    status = completion_status(MIN_EP, completed, step, MAX_STEPS)
    require_unchanged_assets(_validation_context)
    status.update(_validation_context)
    if not status["validation_complete"]:
        atomic_json(os.path.join(out_dir, "validation_status.json"), status)
        print(f"[ERROR] Incomplete validation: {completed}/{MIN_EP} episodes after {step} steps.", flush=True)
        try:
            env.close()
        finally:
            # Kit must release its framework before interpreter exit. Set its native
            # return code too, since shutdown can terminate before SystemExit runs.
            simulation_app.app.post_uncancellable_quit(2)
            try:
                simulation_app.close(wait_for_replicator=False)
            finally:
                sys.exit(2)



    # ─── 8. Agregasi ─────────────────────────────────────────────────────────────
    def _stats(vals):
        a = np.asarray(vals, dtype=np.float64)
        if a.size == 0:
            return {k: float("nan") for k in ("mean", "std", "median", "p90", "p95", "min", "max")}
        return {
            "mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
        }


    n_ep = len(rec_exy)
    exy_mm = _stats([v * 1000 for v in rec_exy])
    ez_mm = _stats([v * 1000 for v in rec_ez])
    norm_mm = _stats([v * 1000 for v in rec_norm])
    mean_ex = float(np.mean(rec_ex)) if rec_ex else float("nan")
    mean_ey = float(np.mean(rec_ey)) if rec_ey else float("nan")
    mean_ez = float(np.mean(rec_ez_s)) if rec_ez_s else float("nan")
    signed_xy_mm = math.sqrt((0 if mean_ex != mean_ex else mean_ex) ** 2
                             + (0 if mean_ey != mean_ey else mean_ey) ** 2) * 1000
    bias_ratio = (signed_xy_mm / exy_mm["mean"]) if exy_mm["mean"] == exy_mm["mean"] and exy_mm["mean"] > 0 else float("nan")
    succ_rate = float(np.mean(rec_succ)) if rec_succ else float("nan")
    reach_rate = float(np.mean(rec_reach)) if rec_reach else float("nan")
    near_fail = float(np.mean(rec_near)) if rec_near else float("nan")
    hold_mean = float(np.mean(rec_hold)) if rec_hold else 0.0

    # Speed and smoothness metrics: EE jerk (m/s³) + control steps to first reach STAY_REQ consecutive
    # in-band steps. stay20_rate = fraction of episodes that ever reached the 20-step stay.
    jerk_stats = _stats(rec_jerk)
    jerk_peak_stats = _stats(rec_jerk_peak)
    steps_stay20_stats = _stats(rec_steps_stay20)
    stay20_rate = (len(rec_steps_stay20) / n_ep) if n_ep else float("nan")

    # Action-fluctuation metrics (deterministic). Read `*_hold` first: the arm is parked
    # in-band for most of an episode, so hold-phase fluctuation is the command motion that
    # does no useful work.
    dact_hold_stats = _stats(rec_dact_hold)
    dact_transit_stats = _stats(rec_dact_transit)
    dact_rev_hold_stats = _stats(rec_dact_rev_hold)
    jvel_hold_stats = _stats(rec_jvel_hold)
    jvel_transit_stats = _stats(rec_jvel_transit)
    jvel_rev_hold_stats = _stats(rec_jvel_rev_hold)


    metrics = {
        **_validation_context,
        "checkpoint": checkpoint_path,
        "checkpoint_tag": checkpoint_tag,
        "num_envs": NUM_ENVS,
        "episodes_completed": n_ep,
        "episodes_requested": MIN_EP,
        "validation_complete": True,
        "steps_run": step,
        "tolerances_m": {"xy": float(GT_SUCCESS_XY_TOL_M), "z": float(GT_SUCCESS_Z_TOL_M)},
        "inference_deterministic": {
            "E_xy_mm": exy_mm, "E_z_mm": ez_mm, "norm_mm": norm_mm,
            "success_rate": succ_rate,
            "success_definition": f"{STAY_REQ} consecutive GT in-band control steps",
            "reach_rate": reach_rate,
            "near_fail_rate": near_fail,
            "precision_hold_steps_mean": hold_mean,
            "bias_signed_xy_mm": signed_xy_mm,
            "bias_signed_components_mm": {"ex": mean_ex * 1000, "ey": mean_ey * 1000, "ez": mean_ez * 1000},
            "bias_ratio_signed_over_Exy": bias_ratio,
            # ── Speed/smoothness/settle metrics ──
            "jerk_mps3": jerk_stats,          # per-episode RMS EE jerk (m/s³)
            "jerk_peak_mps3": jerk_peak_stats,
            "steps_to_stay20": steps_stay20_stats,   # control steps from episode start
            "stay20_rate": stay20_rate,
            "stay_steps_required": STAY_REQ,
            "control_dt_s": DT,
            # ── Hold-phase action fluctuation + joint activity (phase-split, kinematic only) ──
            "dact_rms_hold": dact_hold_stats,        # RMS |Δa| while parked in-band
            "dact_rms_transit": dact_transit_stats,  # RMS |Δa| while approaching
            "dact_reversal_rate_hold": dact_rev_hold_stats,   # 0..1 axes flipping direction/step
            "joint_vel_rms_hold": jvel_hold_stats,           # rad/s over the 6 arm joints
            "joint_vel_rms_transit": jvel_transit_stats,
            "joint_vel_reversal_rate_hold": jvel_rev_hold_stats,
            "motor_load_joints": list(_arm_joint_names),
        },
    }
    require_valid_metrics(metrics)
    atomic_json(os.path.join(out_dir, "inference_metrics.json"), metrics)

    # Per-episode raw arrays (mm) → lets plot_validation.py regenerate figures without
    # re-running the sim.
    np.savez(
        os.path.join(out_dir, "inference_records.npz"),
        exy_mm=np.asarray([v * 1000 for v in rec_exy], dtype=np.float64),
        ez_mm=np.asarray([v * 1000 for v in rec_ez], dtype=np.float64),
        norm_mm=np.asarray([v * 1000 for v in rec_norm], dtype=np.float64),
        ex_mm=np.asarray([v * 1000 for v in rec_ex], dtype=np.float64),
        ey_mm=np.asarray([v * 1000 for v in rec_ey], dtype=np.float64),
        ez_signed_mm=np.asarray([v * 1000 for v in rec_ez_s], dtype=np.float64),
        succ=np.asarray(rec_succ, dtype=np.float64),
        reach=np.asarray(rec_reach, dtype=np.float64),
        hold=np.asarray(rec_hold, dtype=np.float64),
        near=np.asarray(rec_near, dtype=np.float64),
        jerk_mps3=np.asarray(rec_jerk, dtype=np.float64),
        jerk_peak_mps3=np.asarray(rec_jerk_peak, dtype=np.float64),
        steps_to_stay20=np.asarray(rec_steps_stay20, dtype=np.float64),
        dact_rms_hold=np.asarray(rec_dact_hold, dtype=np.float64),
        dact_rms_transit=np.asarray(rec_dact_transit, dtype=np.float64),
        dact_reversal_rate_hold=np.asarray(rec_dact_rev_hold, dtype=np.float64),
        joint_vel_rms_hold=np.asarray(rec_jvel_hold, dtype=np.float64),
        joint_vel_rms_transit=np.asarray(rec_jvel_transit, dtype=np.float64),
        joint_vel_reversal_rate_hold=np.asarray(rec_jvel_rev_hold, dtype=np.float64),
    )
    require_unchanged_assets(_validation_context)
    atomic_json(os.path.join(out_dir, "validation_status.json"), status)


    def _fmt(x, unit="mm"):
        return f"{x:.3f} {unit}" if x == x else "  n/a"


    lines = []
    lines.append(f"Inference validation — {checkpoint_tag}")
    lines.append("=" * 64)
    lines.append(f"Checkpoint : {checkpoint_path}")
    lines.append(f"num_envs   : {NUM_ENVS} parallel | episodes: {n_ep} | steps: {step}")
    lines.append(f"Success tol: xy={GT_SUCCESS_XY_TOL_M*1000:.1f}mm  z={GT_SUCCESS_Z_TOL_M*1000:.1f}mm")
    lines.append("")
    lines.append("INFERENCE metrics (deterministic, mean over episodes):")
    lines.append(f"  {'E_xy':<22}{_fmt(exy_mm['mean'])}")
    lines.append(f"  {'E_z':<22}{_fmt(ez_mm['mean'])}")
    lines.append(f"  {'task norm':<22}{_fmt(norm_mm['mean'])}")
    lines.append(f"  {'stable success rate':<22}{_fmt(succ_rate,'')}  ({STAY_REQ} consecutive steps)")
    lines.append(f"  {'reach rate':<22}{_fmt(reach_rate,'')}  (>=1 in-band step)")
    lines.append(f"  {'precision hold (steps)':<22}{_fmt(hold_mean,'')}")
    lines.append(f"  {'EE jerk RMS (m/s^3)':<22}{_fmt(jerk_stats['mean'],'')}")
    lines.append(f"  {'steps to stay-'+str(STAY_REQ):<22}{_fmt(steps_stay20_stats['mean'],'')}  (rate {_fmt(stay20_rate,'')})")
    lines.append("")
    lines.append("HOLD-PHASE ACTION FLUCTUATION (deterministic; hold = parked in-band, transit = approaching):")
    lines.append(f"  {'|da| RMS hold':<26}{dact_hold_stats['mean']:.5f}      {'|da| RMS transit':<20}{dact_transit_stats['mean']:.5f}")
    lines.append(f"  {'joint vel RMS hold':<26}{jvel_hold_stats['mean']:.5f} rad/s  {'transit':<20}{jvel_transit_stats['mean']:.5f} rad/s")
    lines.append(f"  {'da reversal rate (hold)':<26}{dact_rev_hold_stats['mean']:.4f}   (0..1 = axes flipping direction per step)")
    lines.append(f"  {'jvel reversal rate (hold)':<26}{jvel_rev_hold_stats['mean']:.4f}   over joints: {', '.join(_arm_joint_names)}")
    lines.append("")
    lines.append("INFERENCE distribution (E_xy):")
    lines.append(f"  mean={_fmt(exy_mm['mean'])}  std={_fmt(exy_mm['std'])}  median={_fmt(exy_mm['median'])}")
    lines.append(f"  p90={_fmt(exy_mm['p90'])}  p95={_fmt(exy_mm['p95'])}  min={_fmt(exy_mm['min'])}  max={_fmt(exy_mm['max'])}")
    lines.append("")
    lines.append("BIAS vs JITTER (deterministic):")
    lines.append(f"  signed xy vector = {signed_xy_mm:.3f} mm  (ex={mean_ex*1000:+.3f}, ey={mean_ey*1000:+.3f}, ez={mean_ez*1000:+.3f} mm)")
    lines.append(f"  unsigned E_xy    = {exy_mm['mean']:.3f} mm")
    _ratio_txt = f"{bias_ratio:.3f}" if bias_ratio == bias_ratio else "n/a"
    lines.append(f"  ratio |signed|/E_xy = {_ratio_txt}  (~1 = pure bias/offset, ~0 = residual jitter)")
    lines.append("")
    lines.append(f"near-but-fail rate = {_fmt(near_fail,'')}")
    summary_txt = "\n".join(lines)
    with open(os.path.join(out_dir, "validation_summary.txt"), "w") as f:
        f.write(summary_txt + "\n")
    print("\n" + summary_txt, flush=True)

    # ── Per-metric inference figures (one file each; regenerable on host) ──
    try:
        from plot_validation import render_all as _render_all
        _figs = _render_all(out_dir)
        print(f"[INFO] {len(_figs)} inference figures saved: {', '.join(_figs)}", flush=True)
    except Exception as e:
        print(f"[WARN] plotting failed (regenerate on the host: python3 plot_validation.py {out_dir}): {e}", flush=True)

    print(f"[DONE] Output → {out_dir}", flush=True)
    env.close()
    simulation_app.close()
