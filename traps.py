import os.path

from mynimize import *
from cpflow import *
from cpflow.matrix_utils import cost_HST
from cpflow.topology import *
from collections import namedtuple

from scipy.stats import unitary_group
from typing import List

from tqdm.auto import tqdm

A = namedtuple('Angles', ['angles'])


def random_angles(key, num_angles):
    """Generates a random set of angles."""
    return random.uniform(key, (num_angles, ), minval=0, maxval=2*jnp.pi)


@dataclass
class ExperimentOptions:
    """Options for the experiment."""
    target_type: str
    ansatz_type: str
    num_qubits: int
    gate_counts: List[int]
    num_targets: int
    num_samples: int
    fixed_unitary: jnp.ndarray = None
    fixed_complexity: int = None
    keep_history: bool = False
    num_gd_iterations: int = 2000

    def __post_init__(self):
        assert self.target_type in ['self_instance', 'random_unitary', 'fixed_unitary', 'cz_fixed_complexity']
        assert self.ansatz_type in ['cz', 'cp']
        if self.target_type == 'fixed_unitary':
            assert self.fixed_unitary is not None
        if self.target_type == 'cz_fixed_complexity':
            assert self.fixed_complexity is not None


class Experiment:
    def __init__(self,
                 experiment_options,
                 name):

        self.options = experiment_options
        self.name = name
        self.results = {}
        self.path = f'results/experiment_{name}'
        if os.path.exists(self.path):
            print('WARNING: Experiment already exists and will be overwritten.')

    def run_particular(self, num_gates, num_target):

        random_seed = num_target
        key, subkey = random.split(random.PRNGKey(random_seed))

        anz = Ansatz(self.options.num_qubits, self.options.ansatz_type, fill_layers(connected_layer(self.options.num_qubits), num_gates))
        if self.options.target_type == 'self_instance':
            angles_target = random_angles(subkey, anz.num_angles)
            u_target = anz.unitary(angles_target)
        elif self.options.target_type == 'random_unitary':
            u_target = unitary_group.rvs(2 ** self.options.num_qubits, random_state=random_seed)
        elif self.options.target_type == 'fixed_unitary':
            u_target = self.options.fixed_unitary
        elif self.options.target_type == 'cz_fixed_complexity':
            num_gates_target = self.options.fixed_complexity
            anz_target = Ansatz(self.options.num_qubits, self.options.ansatz_type, fill_layers(connected_layer(self.options.num_qubits), num_gates_target))
            angles_target = random_angles(subkey, anz_target.num_angles)
            u_target = anz_target.unitary(angles_target)

        key, subkey = random.split(key)
        initial_angles = random_angles(subkey, self.options.num_samples * anz.num_angles).reshape((self.options.num_samples, anz.num_angles))
        initial_angles = [A(angles) for angles in initial_angles]
        loss = lambda a: cost_HST(u_target, anz.unitary(a.angles))

        opt_options = OptOptions(num_iterations=self.options.num_gd_iterations, learning_rate=0.1, keep_history=self.options.keep_history)
        result = mynimize(
            loss,
            initial_angles,
            opt_options)

        if not self.options.keep_history:
            del result.all_results
        self.results[num_gates, num_target] = result

    def run(self):
        for num_gates in tqdm(self.options.gate_counts, desc='Gates'):
            for num_target in tqdm(range(self.options.num_targets), desc='Targets', leave=False):
                self.run_particular(num_gates, num_target)
        self.save()

    @staticmethod
    def success_ratio_from_losses(losses, threshold_loss=1e-4, min_loss=0):
        if min_loss == 'empiric':
            min_loss = min(losses)
        return jnp.sum(jnp.array(losses) < min_loss+threshold_loss) / len(losses)

    def get_success_ratios(self, threshold_loss=1e-4, min_loss=None):
        if min_loss is None:
            if self.options.target_type == 'self_instance':
                min_loss = 0
            else:
                min_loss = 'empiric'
        success_ratios = {key: self.success_ratio_from_losses(res.all_best_losses, threshold_loss, min_loss) for key, res in self.results.items()}
        return success_ratios

    def get_mean_and_std_success_ratios(self, threshold_loss=1e-4, min_loss=None):
        success_ratios = self.get_success_ratios(threshold_loss, min_loss)
        mean_success_ratios = []
        std_success_ratios = []
        for num_gates in self.options.gate_counts:
            success_ratios_at_num_gates = jnp.array([success_ratios[num_gates, num_target] for num_target in range(self.options.num_targets)])
            mean_success_ratios.append(jnp.mean(success_ratios_at_num_gates))
            std_success_ratios.append(jnp.std(success_ratios_at_num_gates))
        return jnp.array(mean_success_ratios), jnp.array(std_success_ratios)

    def plot_success_ratios(self, threshold_loss=1e-4, min_loss=None, maketitle=True, label='', xshift=0):

        mean_success_ratios, std_success_ratios = self.get_mean_and_std_success_ratios(threshold_loss, min_loss)
        gate_counts = jnp.array(self.options.gate_counts)+xshift
        plt.errorbar(
            gate_counts,
            mean_success_ratios,
            yerr=std_success_ratios,
            fmt='o',
            capsize=10,
            label=label);

        if maketitle:
            plt.title(f'{self.options.target_type} {self.options.ansatz_type} {self.options.num_qubits}q')
        plt.yscale('log')
        plt.xticks(self.options.gate_counts, rotation=70)
        plt.ylabel('Success Ratio')
        plt.xlabel('Number of 2q gates')

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            results = dill.load(f)

        return results


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


# for ansatz_type in ['cz', 'cp']:
#     options = ExperimentOptions(
#         target_type='fixed_unitary',
#         num_qubits=4,
#         num_gd_iterations=5000,
#         num_targets=1,
#         fixed_unitary=jnp.identity(2**4)
#     )
#     experiment = Experiment(options, f'{ansatz_type}_id_4q_5000gd')
#     experiment.run()