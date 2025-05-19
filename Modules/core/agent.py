from core.neural_network import MLModel
from jax import jit, random, lax, value_and_grad, vmap
import jax.numpy as jnp
import optax
from functools import partial


class DQN():
    def __init__(
        self,
        model: MLModel,
        discount: float,
        num_data_qubits: int,
    ) -> None:
        """
        Deep Q-learning agent.

        model: The model that will be used for the policy decision making

        discount: The discount of future rewards

        n_actions: Number of possible action the agent can take
        """
        self.model = model
        self.discount = discount
        self.num_data_qubits = num_data_qubits
        self.num_deformations = 6

    def act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        epsilon: float,
    ):
        """
        Let the agent take a decision based on the system state.

        online_net_params: The parameters for the agents policy neural network

        state: Array of int's denoting the current state of the system

        disallowed_actions: Array of bool denoting which action are not allowed to be taken by the agent

        epsilon: The probability that the agent will take a random allowed action

        returns: Tuple of (actions, done, key)
        """
        return self._act(key, online_net_params, state, epsilon)

    @partial(jit, static_argnames=("self"))
    def _act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        epsilon: float,
    ):
        def _random_action(args):
            subkey, invariant_actions = args
            rv = random.uniform(
                subkey, 
                shape=invariant_actions.shape
            ) - (invariant_actions == True)
            return rv.argmax()

        def _policy_action(args):
            subkey, invariant_actions = args
            q_values = self.model.apply_single(online_net_params, state).flatten()
            q_values = jnp.where(invariant_actions, -jnp.inf, q_values)
            desired_action = jnp.argmax(q_values)
            return desired_action

        # The actions that leave the deformation the same
        invariant_actions = jnp.zeros(
            self.num_data_qubits*self.num_deformations, 
            dtype=jnp.bool
        ).at[
            self.merge_action(state, jnp.arange(self.num_data_qubits))
        ].set(True)

        explore = random.uniform(key) < epsilon
        key, subkey = random.split(key)
        action = lax.cond(
            explore,
            _random_action,
            _policy_action,
            operand=(subkey, invariant_actions),
        )
        
        return action, subkey

    def split_action(
        self,
        action: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Splits an action index into it's components, deformation index and data qubit index respectively
        """
        return jnp.unravel_index(
            action,
            shape=(
                self.num_deformations,
                self.num_data_qubits
            )
        )

    def merge_action(
        self,
        deformation_action_idx: jnp.ndarray,
        data_qubit_action_idx: jnp.ndarray,
    ):
        """
        Combines the split action back into a single action index - the inverse of split action
        """
        return deformation_action_idx*self.num_data_qubits + data_qubit_action_idx

    @partial(jit, static_argnames=("self", "optimizer"))
    def update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[
            str: jnp.ndarray
        ],  # state, score, action, reward, next_state, done
    ):
        @jit
        def _batch_loss_fn(
            online_net_params: dict,
            target_net_params: dict,
            states: jnp.ndarray,
            scores: jnp.ndarray,
            actions: jnp.ndarray,
            rewards: jnp.ndarray,
            next_states: jnp.ndarray,
            dones: jnp.ndarray,
        ):
            # vectorize the loss over states, actions, rewards, next_states and done flags
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
            def _loss_fn(
                online_net_params,
                target_net_params,
                state,
                score,
                action,
                reward,
                next_state,
                done,
            ):
                target = reward + (1 - done) * self.discount * jnp.max(
                    self.model.apply_single(target_net_params, next_state),
                )
                prediction = self.model.apply_single(
                    online_net_params, state).flatten()[action]
                return jnp.square(target - prediction)

            return jnp.mean(
                _loss_fn(
                    online_net_params,
                    target_net_params,
                    states,
                    scores,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ),
                axis=0,
            )

        loss, grads = value_and_grad(_batch_loss_fn)(
            online_net_params, target_net_params, **experiences
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        online_net_params = optax.apply_updates(online_net_params, updates)

        return online_net_params, optimizer_state, loss
