import numpy as np


#controllet itself

def pd_controller(kp, kd, q, qdes, qvel):
    return kp * (qdes - q) - kd * qvel

def predict_step(q, dq, v, dt):
    dq_next = dq + v * dt
    q_next  = q + dq * dt
    return q_next, dq_next

def stage_cost(q, dq, q_des, v):
    w_q  = 20.0   # точность
    w_dq = 1.0    # демпфирование
    w_v  = 0.1    # не дёргать приводы

    return (
        w_q  * np.sum((q - q_des)**2)
        + w_dq * np.sum(dq**2)
        + w_v  * np.sum(v**2)
    )

def rollout_cost(q0, dq0, v_seq, q_des, dt):
    q = q0.copy()
    dq = dq0.copy()
    cost = 0.0

    for v in v_seq:
        q, dq = predict_step(q, dq, v, dt)
        cost += stage_cost(q, dq, q_des, v)

    return cost

def mpc_step(q, dq, q_des, dt):
    N = 5
    v_candidates = [-2.0, 0.0, 2.0]

    best_cost = np.inf
    best_v0 = np.zeros_like(q)

    for v0 in v_candidates:
        for v1 in v_candidates:
            for v2 in v_candidates:
                for v3 in v_candidates:
                    for v4 in v_candidates:

                        v_seq = [
                            v0 * np.ones_like(q),
                            v1 * np.ones_like(q),
                            v2 * np.ones_like(q),
                            v3 * np.ones_like(q),
                            v4 * np.ones_like(q),
                        ]

                        cost = rollout_cost(q, dq, v_seq, q_des, dt)

                        if cost < best_cost:
                            best_cost = cost
                            best_v0 = v_seq[0]

    return best_v0