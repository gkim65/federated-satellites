import numpy as np
import wandb
import os
import pandas as pd
import shutil

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
                       waterfall_step,
                       waterfall_phase):

    start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
    if start_time_og == 0:
        start_time_og = start_time_sec

    cluster_clients = []

    # Hard code the initial value that the sims start with
    if all(sim_time == 0 for sim_time in sim_times_currents):
        for i in range(len(sim_times_currents)):
            sim_times_currents[i] = 1711987200
    # -------------------------------------------------------
    # WATERFALL AGGREGATION PHASES
    # Only enter if all clusters have finished local training
    # -------------------------------------------------------
    if all(i >= start_time_og for i in sim_times_currents):

        if agg_true:
            mid = cluster_n // 2

            # -------------------
            # PHASE 1: SCATTER
            # -------------------
            if waterfall_phase == "scatter":

                if waterfall_step < mid:
                    # Left side: plane (step+1) -> (step+2)
                    # Right side: plane (P-step) -> (P-step-1)
                    left_pair  = (waterfall_step + 1, waterfall_step + 2)
                    right_pair = (cluster_n - waterfall_step, cluster_n - waterfall_step - 1)

                    start_time, end_time, counter, epoch_train, idle_time, duration_round = \
                        scheduleAdjacentISL(sat_df, counter, factor_c,
                                            start_time_og, int(config_epochs) + 120,
                                            left_pair, right_pair)

                    wandb.log({"waterfall_step": waterfall_step,
                               "waterfall_phase": "scatter",
                               "left_pair_send": left_pair[0],
                               "left_pair_recv": left_pair[1],
                               "right_pair_send": right_pair[0],
                               "right_pair_recv": right_pair[1],
                               "server_round": server_round,
                               "start_time": start_time,
                               "end_time": end_time,
                               "idle_time": idle_time})

                    start_time_og = start_time
                    epochs = epoch_train
                    waterfall_step += 1

                    # Decide next phase
                    if waterfall_step >= mid:
                        if cluster_n % 2 == 0:
                            waterfall_phase = "middle_exchange"
                        else:
                            # Odd P: middle plane already has full model
                            waterfall_phase = "allgather"
                            waterfall_step = mid - 1

                    # Receiving planes for this scatter step
                    left_recv  = left_pair[1]
                    right_recv = right_pair[1]
                    recv_clients = []
                    for c in range(sat_n):
                        recv_clients.append(sat_n * (left_recv - 1) + c)
                        recv_clients.append(sat_n * (right_recv - 1) + c)

                    x = [[client, left_recv, left_recv] for client in clients
                         if int(client.partition_id) in recv_clients]
                    y = [[clients[i], i, i] for i in range(1, cluster_n + 1)]
                    agg_true = False

                    print("FIRST RETURNNNN")
                    return (x, y, counter, sim_times_currents,
                            cluster_round_starts, cluster_round_currents,
                            "waterfall_scatter", epochs, start_time_og,
                            agg_true, waterfall_step, waterfall_phase)

            # ----------------------------
            # PHASE 2: MIDDLE EXCHANGE
            # (even P only)
            # ----------------------------
            elif waterfall_phase == "middle_exchange":
                mid_left  = cluster_n // 2
                mid_right = cluster_n // 2 + 1
                mid_pair  = (mid_left, mid_right)

                start_time, end_time, counter, epoch_train, idle_time, duration_round = \
                    scheduleAdjacentISL(sat_df, counter, factor_c,
                                        start_time_og, int(config_epochs) + 120,
                                        mid_pair, None)

                wandb.log({"waterfall_phase": "middle_exchange",
                           "mid_left": mid_left,
                           "mid_right": mid_right,
                           "server_round": server_round,
                           "start_time": start_time,
                           "end_time": end_time})

                start_time_og  = start_time
                epochs         = epoch_train
                waterfall_phase = "allgather"
                waterfall_step  = cluster_n // 2 - 1

                mid_clients = []
                for c in range(sat_n):
                    mid_clients.append(sat_n * (mid_left  - 1) + c)
                    mid_clients.append(sat_n * (mid_right - 1) + c)

                x = [[client, mid_left, mid_left] for client in clients
                     if int(client.partition_id) in mid_clients]
                y = [[clients[i], i, i] for i in range(1, cluster_n + 1)]
                agg_true = False

                print("SECOND RETURNNNN")
                return (x, y, counter, sim_times_currents,
                        cluster_round_starts, cluster_round_currents,
                        "middle_exchange", epochs, start_time_og,
                        agg_true, waterfall_step, waterfall_phase)

            # -------------------
            # PHASE 3: ALLGATHER
            # -------------------
            elif waterfall_phase == "allgather":

                if waterfall_step >= 1:
                    # Propagate global model outward from middle
                    left_pair  = (waterfall_step + 1, waterfall_step)
                    right_pair = (cluster_n - waterfall_step, cluster_n - waterfall_step + 1)

                    start_time, end_time, counter, epoch_train, idle_time, duration_round = \
                        scheduleAdjacentISL(sat_df, counter, factor_c,
                                            start_time_og, int(config_epochs) + 120,
                                            left_pair, right_pair)

                    wandb.log({"waterfall_step": waterfall_step,
                               "waterfall_phase": "allgather",
                               "left_pair_send": left_pair[0],
                               "left_pair_recv": left_pair[1],
                               "right_pair_send": right_pair[0],
                               "right_pair_recv": right_pair[1],
                               "server_round": server_round,
                               "start_time": start_time,
                               "end_time": end_time})

                    start_time_og = start_time
                    epochs        = epoch_train
                    waterfall_step -= 1

                    left_recv  = left_pair[1]
                    right_recv = right_pair[1]
                    recv_clients = []
                    for c in range(sat_n):
                        recv_clients.append(sat_n * (left_recv  - 1) + c)
                        recv_clients.append(sat_n * (right_recv - 1) + c)

                    x = [[client, left_recv, left_recv] for client in clients
                         if int(client.partition_id) in recv_clients]
                    y = [[clients[i], i, i] for i in range(1, cluster_n + 1)]
                    agg_true = False

                    # Allgather complete — reset for next FL round
                    if waterfall_step < 1:
                        waterfall_phase = "scatter"
                        waterfall_step  = 0

                   
                    print("Third? RETURNNNN")
                    return (x, y, counter, sim_times_currents,
                            cluster_round_starts, cluster_round_currents,
                            "waterfall_allgather", epochs, start_time_og,
                            agg_true, waterfall_step, waterfall_phase)

    # -------------------------------------------------------
    # LOCAL TRAINING
    # Train the cluster that is furthest behind
    # -------------------------------------------------------


    """
    Normal fL stuff:
    """
    print(sim_times_currents)
    print(start_time_og)
    
    # if anything is less than the current start_time set
    if any(i <= start_time_og for i in sim_times_currents):

        cluster_index = np.argmin(cluster_round_currents)
        cluster       = cluster_index + 1
        cluster_round_currents[cluster_index] += 1
        sim_times_currents[cluster_index]     += epochs

        if all(i >= start_time_og for i in sim_times_currents):
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

        print("4th RETURNNNN")
        return (x, x, counter, sim_times_currents,
                cluster_round_starts, cluster_round_currents,
                "local_cluster", epochs, start_time_og,
                agg_true, waterfall_step, waterfall_phase)


# -------------------------------------------------------
# ADJACENT ISL SCHEDULER
# Only looks for the specific adjacent pair(s) needed
# at each waterfall step — never selects non-adjacent pairs
# -------------------------------------------------------
def scheduleAdjacentISL(sat_df, counter, factor_c, start_time_og,
                        epochs, pair_left, pair_right):
    """
    Find the next available inter-SL window for adjacent plane pairs only.

    Args:
        pair_left:  (plane_a, plane_b) for left side of waterfall
        pair_right: (plane_a, plane_b) for right side, or None for single pair

    Raises:
        ValueError if no valid window found before end of sat_df
    """
    count_temp   = counter
    found_left   = False
    found_right  = pair_right is None  # if no right pair needed mark as done

    left_start,  left_end  = 0, 0
    right_start, right_end = 0, 0

    training_complete_time = start_time_og + epochs

    while not (found_left and found_right):

        # Safety: stop if we run off end of dataframe
        if count_temp >= len(sat_df):
            raise ValueError(
                f"No valid inter-SL window found for pairs "
                f"left={pair_left}, right={pair_right}. "
                f"Searched {count_temp - counter} rows from counter={counter}. "
                f"Training completes at t={training_complete_time:.0f}. "
                f"Last row time="
                f"{sat_df['Start Time Seconds Cumulative'].iloc[-1]:.0f}."
            )

        cluster_id_1 = int(sat_df['cluster_num_1'].iloc[count_temp] / factor_c)
        cluster_id_2 = int(sat_df['cluster_num_2'].iloc[count_temp] / factor_c)

        # Normalize order for lookup
        if cluster_id_1 > cluster_id_2:
            cluster_id_1, cluster_id_2 = cluster_id_2, cluster_id_1

        temp_start = sat_df['Start Time Seconds Cumulative'].iloc[count_temp]
        temp_end   = sat_df['End Time Seconds Cumulative'].iloc[count_temp]
        duration   = sat_df['Duration (sec)'].iloc[count_temp]

        training_done      = temp_start > training_complete_time
        window_long_enough = duration > 200

        if training_done and window_long_enough:

            if not found_left and pair_left is not None:
                l1, l2 = min(pair_left), max(pair_left)
                if cluster_id_1 == l1 and cluster_id_2 == l2:
                    left_start, left_end = temp_start, temp_end
                    found_left = True
                    print(f"  ✓ Left pair {pair_left} @ t={temp_start:.0f} "
                          f"(dur={duration:.0f}s)")

            if not found_right and pair_right is not None:
                r1, r2 = min(pair_right), max(pair_right)
                if cluster_id_1 == r1 and cluster_id_2 == r2:
                    right_start, right_end = temp_start, temp_end
                    found_right = True
                    print(f"  ✓ Right pair {pair_right} @ t={temp_start:.0f} "
                          f"(dur={duration:.0f}s)")

        count_temp += 1

    # Use the later start so both sides are ready simultaneously
    new_start_time = max(left_start, right_start) if pair_right else left_start
    new_end_time   = max(left_end,   right_end)   if pair_right else left_end

    # Sanity checks
    assert new_start_time > training_complete_time, (
        f"Selected window t={new_start_time:.0f} is before "
        f"training completes at t={training_complete_time:.0f}"
    )
    assert new_end_time > new_start_time, (
        f"Invalid window: end={new_end_time:.0f} <= start={new_start_time:.0f}"
    )

    idle_time      = abs(left_start - right_start) if pair_right else 0
    epoch_train    = new_start_time - start_time_og
    duration_round = new_end_time - new_start_time

    print(f"  Window: start={new_start_time:.0f}, end={new_end_time:.0f}, "
          f"dur={duration_round:.0f}s, idle={idle_time:.0f}s")

    return new_start_time, new_end_time, count_temp, epoch_train, idle_time, duration_round