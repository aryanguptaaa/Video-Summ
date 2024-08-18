import numpy as np


def eval_metrics(y_pred, y_true):

    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return [precision, recall, fscore]


def select_keyshots(video_info, pred_score):

    vidlen = video_info['length'][()]
    cps = video_info['change_points'][:]
    weight = video_info['n_frame_per_seg'][:]
    pred_score = np.array(pred_score)
    pred_score = upsample(pred_score, vidlen)
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    
    # Use PSO for selection
    selected = pso_selection(pred_value, weight, int(0.15 * vidlen))
    selected = selected[::-1]
    key_labels = np.zeros((vidlen,))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1

    return pred_score.tolist(), selected, key_labels.tolist()


def upsample(down_arr, vidlen):

    up_arr = np.zeros(vidlen)
    ratio = vidlen // 320
    l = (vidlen - ratio * 320) // 2
    i = 0
    while i < len(down_arr):  # Use len(down_arr) to check valid index
        up_arr[l:l + ratio] = np.ones(ratio, dtype=int) * down_arr[i]
        l += ratio
        i += 1

    return up_arr


def pso_selection(values, weights, capacity):
    # Define PSO parameters
    num_particles = 30
    num_iterations = 100
    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5

    # Initialize particles
    particles = np.random.randint(2, size=(num_particles, len(values)))
    velocities = np.random.rand(num_particles, len(values)) * 2 - 1
    personal_best_positions = np.copy(particles)
    personal_best_scores = np.zeros(num_particles)

    for i in range(num_particles):
        personal_best_scores[i] = fitness_function(personal_best_positions[i], values, weights, capacity)

    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = np.max(personal_best_scores)

    for iteration in range(num_iterations):
        for i in range(num_particles):
            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_weight * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                             social_weight * np.random.rand() * (global_best_position - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], 0, 1)
            score = fitness_function(particles[i], values, weights, capacity)

            if score > personal_best_scores[i]:
                personal_best_positions[i] = np.copy(particles[i])
                personal_best_scores[i] = score

                if score > global_best_score:
                    global_best_position = np.copy(particles[i])
                    global_best_score = score

    return np.where(global_best_position > 0.5)[0]


def fitness_function(selection, values, weights, capacity):
    total_value = np.dot(selection, values)
    total_weight = np.dot(selection, weights)

    if total_weight > capacity:
        return 0
    else:
        return total_value
