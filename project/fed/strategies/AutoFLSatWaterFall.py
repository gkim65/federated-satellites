import numpy as np
import wandb


# -------------------------------------------------------
# SEQUENCE BUILDER
# Generates the flat ordered list of waterfall steps
# for any cluster count P >= 2
# -------------------------------------------------------

def build_waterfall_sequence(cluster_n):
    """
    Build a flat ordered list of (phase, left_pair, right_pair) steps.

    Examples:
        P=2: [('middle_exchange', (1,2), None),
              ('allgather',       (2,1), None)]

        P=3: [('scatter',   (1,2), (3,2)),
              ('allgather', (2,1), (2,3))]

        P=4: [('scatter',         (1,2), (4,3)),
              ('middle_exchange', (2,3), None),
              ('allgather',       (2,1), (3,4))]

        P=5: [('scatter',   (1,2), (5,4)),
              ('scatter',   (2,3), (4,3)),
              ('allgather', (3,2), (3,4)),
              ('allgather', (2,1), (4,5))]
    """
    sequence = []
    mid = cluster_n // 2

    if cluster_n == 2:
        sequence.append(("middle_exchange", (1, 2), None))
        sequence.append(("allgather",       (2, 1), None))
        return sequence

    # --- Scatter phase ---
    for s in range(mid):
        left_pair  = (s + 1, s + 2)
        right_pair = (cluster_n - s, cluster_n - s - 1)
        # For odd P at the middle step, left and right are the same pair — drop right
        if (min(left_pair) == min(right_pair) and
                max(left_pair) == max(right_pair)):
            right_pair = None
        sequence.append(("scatter", left_pair, right_pair))

    # --- Middle exchange (even P only) ---
    if cluster_n % 2 == 0:
        mid_left  = cluster_n // 2
        mid_right = cluster_n // 2 + 1
        sequence.append(("middle_exchange", (mid_left, mid_right), None))

    # --- Allgather phase ---
    for s in range(mid - 1, -1, -1):
        left_pair  = (s + 2, s + 1)                        # reverse of scatter left
        right_pair = (cluster_n - s - 1, cluster_n - s)    # reverse of scatter right
        if (min(left_pair) == min(right_pair) and
                max(left_pair) == max(right_pair)):
            right_pair = None
        sequence.append(("allgather", left_pair, right_pair))

    return sequence


# -------------------------------------------------------
# MAIN ALGORITHM
# -------------------------------------------------------

def AutoFLSat_Waterfall(sat_df,
                       counter,
                       client_n,
                       client_max,
                       sat_n,
                       cluster_n,
                       factor_s,
                       factor_c,
                       server_round,
                       clients,
                       name,
                       alg,
                       config_epochs,
                       sim_times_currents,
                       cluster_round_starts,
                       cluster_round_currents,
                       epochs,
                       start_time_og,
                       agg_true,
                       waterfall_seq_index,
                       waterfall_sequence,
                       dropout_rate=0.0):

    cluster_clients = []

    # Hard code the initial value that the sims start with
    if all(sim_time == 0 for sim_time in sim_times_currents):
        for i in range(len(sim_times_currents)):
            sim_times_currents[i] = 1711987200

    if start_time_og == 0:
        start_time_og = sim_times_currents[0]

    # Build sequence if empty (first round or after reset)
    if not waterfall_sequence:
        waterfall_sequence = build_waterfall_sequence(cluster_n)
        print(f"Waterfall sequence for {cluster_n} clusters:")
        for i, step in enumerate(waterfall_sequence):
            print(f"  {i}: {step}")

    # Log state every round
    phase_map = {"scatter": 1, "middle_exchange": 2, "allgather": 3}
    current_phase = waterfall_sequence[waterfall_seq_index][0] if waterfall_sequence else "local"
    wandb.log({"waterfall_phase": phase_map.get(current_phase, 0) if agg_true else 0,
               "waterfall_seq_index": waterfall_seq_index,
               "agg_true": int(agg_true),
               "start_time_og": start_time_og,
               "sim_times_min": min(sim_times_currents),
               "sim_times_max": max(sim_times_currents),
               "server_round": server_round})

    # -------------------------------------------------------
    # WATERFALL AGGREGATION
    # -------------------------------------------------------
    if all(i >= start_time_og for i in sim_times_currents) and agg_true:

        # Get current step from sequence
        phase, left_pair, right_pair = waterfall_sequence[waterfall_seq_index]

        start_time, end_time, counter, epoch_train, idle_time, duration_round = \
            scheduleAdjacentISL(sat_df, counter, factor_c,
                                start_time_og, int(config_epochs) + 120,
                                left_pair, right_pair,
                                dropout_rate=dropout_rate)

        wandb.log({"waterfall_phase_name": phase,
                   "waterfall_seq_index": waterfall_seq_index,
                   "left_pair_0": left_pair[0],
                   "left_pair_1": left_pair[1],
                   "server_round": server_round,
                   "start_time": start_time,
                   "end_time": end_time,
                   "idle_time": idle_time})

        start_time_og = start_time
        epochs        = epoch_train

        # Determine which planes receive updates this step
        left_recv    = left_pair[1]
        recv_clients = []
        for c in range(sat_n):
            recv_clients.append(sat_n * (left_recv - 1) + c)

        if right_pair is not None:
            right_recv = right_pair[1]
            for c in range(sat_n):
                recv_clients.append(sat_n * (right_recv - 1) + c)

        x = [[client, left_recv, left_recv] for client in clients
             if int(client.partition_id) in recv_clients]
        y = [[clients[i], i, i] for i in range(1, cluster_n + 1)]

        # Advance sequence
        waterfall_seq_index += 1

        # Check if waterfall complete
        if waterfall_seq_index >= len(waterfall_sequence):
            print(f"Waterfall complete — resetting for next FL round")
            waterfall_seq_index = 0
            waterfall_sequence  = []  # will rebuild next round
            # Reset timing so local training fires correctly next round
            start_time_og = max(sim_times_currents)
            for i in range(len(sim_times_currents)):
                sim_times_currents[i] = start_time_og
            agg_true = False

        return (x, y, counter, sim_times_currents,
                cluster_round_starts, cluster_round_currents,
                phase, epochs, start_time_og,
                agg_true, waterfall_seq_index, waterfall_sequence)

    # -------------------------------------------------------
    # LOCAL TRAINING
    # Train the cluster that is furthest behind
    # -------------------------------------------------------
    if any(i <= start_time_og for i in sim_times_currents):

        cluster_index = np.argmin(cluster_round_currents)
        cluster       = cluster_index + 1
        cluster_round_currents[cluster_index] += 1
        sim_times_currents[cluster_index]     += epochs

        if all(i > start_time_og for i in sim_times_currents):
            agg_true = True

        wandb.log({"cluster_round_current_time": sim_times_currents[cluster_index],
                   "cluster_id": cluster,
                   "cluster_rounds": cluster_round_currents[cluster_index],
                   "agg_type": "localAgg",
                   "server_round": server_round})

        for client in range(sat_n):
            client_id = sat_n * cluster - (sat_n - client)
            cluster_clients.append(client_id)

        x = [[client, cluster, cluster] for client in clients
             if int(client.partition_id) in cluster_clients]

        return (x, x, counter, sim_times_currents,
                cluster_round_starts, cluster_round_currents,
                "local_cluster", epochs, start_time_og,
                agg_true, waterfall_seq_index, waterfall_sequence)

    # Safety fallback — should never reach here
    print(f"WARNING: AutoFLSatWaterfall fell through all conditions.")
    print(f"  agg_true={agg_true}, start_time_og={start_time_og}")
    print(f"  sim_times_currents={sim_times_currents}")
    print(f"  waterfall_seq_index={waterfall_seq_index}")

    # Force reset and retrain
    start_time_og = max(sim_times_currents)
    for i in range(len(sim_times_currents)):
        sim_times_currents[i] = start_time_og

    cluster_index = np.argmin(cluster_round_currents)
    cluster       = cluster_index + 1
    cluster_round_currents[cluster_index] += 1
    sim_times_currents[cluster_index]     += epochs

    for client in range(sat_n):
        client_id = sat_n * cluster - (sat_n - client)
        cluster_clients.append(client_id)

    x = [[client, cluster, cluster] for client in clients
         if int(client.partition_id) in cluster_clients]

    return (x, x, counter, sim_times_currents,
            cluster_round_starts, cluster_round_currents,
            "local_cluster", epochs, start_time_og,
            agg_true, waterfall_seq_index, waterfall_sequence)


# -------------------------------------------------------
# ADJACENT ISL SCHEDULER
# -------------------------------------------------------

def scheduleAdjacentISL(sat_df, counter, factor_c, start_time_og,
                        epochs, pair_left, pair_right, dropout_rate=0.0):
    """
    Find the next available inter-SL window for adjacent plane pairs only.
    """
    count_temp    = counter
    found_left    = False
    found_right   = pair_right is None

    left_start,  left_end  = 0, 0
    right_start, right_end = 0, 0

    dropped_left  = 0
    dropped_right = 0

    training_complete_time = start_time_og + epochs

    while not (found_left and found_right):

        if count_temp >= len(sat_df):
            raise ValueError(
                f"No valid inter-SL window found for pairs "
                f"left={pair_left}, right={pair_right}. "
                f"Searched {count_temp - counter} rows from counter={counter}. "
                f"Dropped {dropped_left} left, {dropped_right} right windows. "
                f"Training completes at t={training_complete_time:.0f}. "
                f"Last row time={sat_df['Start Time Seconds Cumulative'].iloc[-1]:.0f}."
            )

        cluster_id_1 = int(sat_df['cluster_num_1'].iloc[count_temp] / factor_c)
        cluster_id_2 = int(sat_df['cluster_num_2'].iloc[count_temp] / factor_c)

        if cluster_id_1 > cluster_id_2:
            cluster_id_1, cluster_id_2 = cluster_id_2, cluster_id_1

        temp_start = sat_df['Start Time Seconds Cumulative'].iloc[count_temp]
        temp_end   = sat_df['End Time Seconds Cumulative'].iloc[count_temp]
        duration   = sat_df['Duration (sec)'].iloc[count_temp]

        training_done      = temp_start > training_complete_time
        window_long_enough = duration > 200

        if training_done and window_long_enough:

            if not found_left:
                l1, l2 = min(pair_left), max(pair_left)
                if cluster_id_1 == l1 and cluster_id_2 == l2:
                    if np.random.random() < dropout_rate:
                        dropped_left += 1
                        print(f"  ✗ Dropped left {pair_left} @ t={temp_start:.0f}")
                    else:
                        left_start, left_end = temp_start, temp_end
                        found_left = True
                        print(f"  ✓ Left {pair_left} @ t={temp_start:.0f} (dur={duration:.0f}s)")

            if not found_right and pair_right is not None:
                r1, r2 = min(pair_right), max(pair_right)
                if cluster_id_1 == r1 and cluster_id_2 == r2:
                    if np.random.random() < dropout_rate:
                        dropped_right += 1
                        print(f"  ✗ Dropped right {pair_right} @ t={temp_start:.0f}")
                    else:
                        right_start, right_end = temp_start, temp_end
                        found_right = True
                        print(f"  ✓ Right {pair_right} @ t={temp_start:.0f} (dur={duration:.0f}s)")

        count_temp += 1

    new_start_time = max(left_start, right_start) if pair_right else left_start
    new_end_time   = max(left_end,   right_end)   if pair_right else left_end

    assert new_start_time > training_complete_time, (
        f"Window t={new_start_time:.0f} before training "
        f"completes t={training_complete_time:.0f}"
    )

    idle_time      = abs(left_start - right_start) if pair_right else 0
    epoch_train    = new_start_time - start_time_og
    duration_round = new_end_time - new_start_time

    wandb.log({"dropped_windows_left":  dropped_left,
               "dropped_windows_right": dropped_right,
               "total_dropped_windows": dropped_left + dropped_right,
               "dropout_rate": dropout_rate})

    print(f"  Window: start={new_start_time:.0f}, dur={duration_round:.0f}s, "
          f"idle={idle_time:.0f}s, dropped={dropped_left + dropped_right}")

    return new_start_time, new_end_time, count_temp, epoch_train, idle_time, duration_round