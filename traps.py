from mynimize import *
from cpflow import *
from cpflow.matrix_utils import cost_HST
from cpflow.topology import *
from collections import namedtuple

from scipy.stats import unitary_group

A = namedtuple('Angles', ['angles'])


def random_angles(key, num_angles):
    return random.uniform(key, (num_angles, ), minval=0, maxval=2*jnp.pi)


def chart_success_ratio(target_type, ansatz_type, num_qubits, gate_counts, num_self_instances, num_samples, threshold_loss=1e-4, fixed_unitary=None):
    key = random.PRNGKey(0)

    success_ratios = {}
    for num_gates in gate_counts:
        success_ratios_at_num_gates = []
        for random_seed in range(num_self_instances):
            key, subkey = random.split(key)
            sr = success_ratio(target_type, ansatz_type, num_qubits, num_gates, num_samples, random_seed, threshold_loss, fixed_unitary)
            success_ratios_at_num_gates.append(sr)
        success_ratios[num_gates] = success_ratios_at_num_gates

    return success_ratios


def success_ratio_from_losses(losses, threshold_loss, min_loss=0):
    if min_loss is None:
        min_loss = min(losses)
    return sum(losses < min_loss+threshold_loss) / len(losses)


def success_ratio(target_type, ansatz_type, num_qubits, num_gates, num_samples, random_seed, threshold_loss, fixed_unitary=None):

    key, subkey = random.split(random.PRNGKey(random_seed))

    anz = Ansatz(num_qubits, ansatz_type, fill_layers(connected_layer(num_qubits), num_gates))
    if target_type == 'self_instance':
        angles_target = random_angles(subkey, anz.num_angles)
        u_target = anz.unitary(angles_target)
        min_loss = 0
    elif target_type == 'random_unitary':
        u_target = unitary_group.rvs(2**num_qubits, random_state=random_seed)
        min_loss = None
    elif target_type == 'fixed_unitary':
        u_target = fixed_unitary
        min_loss = None
    else:
        raise ValueError('target_type must be one of "self_instance", "random_unitary", "fixed_unitary"')

    key, subkey = random.split(key)
    initial_angles = random_angles(subkey, num_samples * anz.num_angles).reshape((num_samples, anz.num_angles))
    initial_angles = [A(angles) for angles in initial_angles]
    loss = lambda a: cost_HST(u_target, anz.unitary(a.angles))

    result = mynimize(loss, initial_angles, OptOptions(num_iterations=2000, learning_rate=0.1))
    result.save('results/%s_%s_%dq_%dd_%d' % (target_type, ansatz_type, num_qubits, num_gates, random_seed))
    best_losses = jnp.array(result.all_best_losses)
    return success_ratio_from_losses(best_losses, threshold_loss, min_loss)


def hard_removal(anz, angles, num_angle, minimizer):
    """
    Sets angle 'num_angle' to zero and attempts to recover the original circuit.
    """

    u_target = anz.unitary(angles)
    loss_func = lambda x: cost_HST(anz.unitary(x), u_target)

    initial_angles = angles.at[num_angle].set(0)
    loss, min_angles = minimizer(loss_func, initial_angles)
    return loss


def soft_removal(anz, angles, num_angle, minimizer, r=1e-2):
    """
    Adds a regularizer to angle 'num_angle' in an attempt to remove it.
    """

    u_target = anz.unitary(angles)
    reg_func = lambda a: r*jnp.abs(a[num_angle])
    loss_func = lambda a: cost_HST(anz.unitary(a), u_target)
    regloss_func = lambda a: loss_func(a) + reg_func(a)

    initial_angles = angles
    loss, min_angles = minimizer(regloss_func, initial_angles)

    return loss_func(min_angles)


