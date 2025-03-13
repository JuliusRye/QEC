from neural_network import MLModel
from jax import jit, random, lax, value_and_grad, vmap
import jax.numpy as jnp
import optax
from functools import partial


class DQN():
    def __init__(
        self,
        model: MLModel,
        discount: float,
        n_actions: int,
    ) -> None:
        """
        Deep Q-learning agent.

        model: The model that will be used for the policy decision making

        discount: The discount of future rewards

        n_actions: Number of possible action the agent can take
        """
        self.model = model
        self.discount = discount
        self.n_actions = n_actions

    @partial(jit, static_argnames=("self"))
    def act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        disallowed_actions: jnp.ndarray,
        epsilon: float,
    ):

        def _random_action(subkey):
            rv = random.uniform(
                subkey, shape=disallowed_actions.shape) * (disallowed_actions == False)
            return rv.argmax(), jnp.all(disallowed_actions)
            # return random.choice(subkey, jnp.arange(self.n_actions)), False

        def _policy_action(_):
            q_values = jnp.where(
                disallowed_actions,
                -jnp.inf,
                self.model.apply_batch(online_net_params, state).flatten()
            )
            # q_values = self.model.apply(online_net_params, state)
            done = jnp.max(q_values) <= 0
            return jnp.argmax(q_values), done

        explore = random.uniform(key) < epsilon
        key, subkey = random.split(key)
        action, done = lax.cond(
            explore,
            _random_action,
            _policy_action,
            operand=subkey,
        )
        return action, done, subkey

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
                    self.model.apply_batch(target_net_params, next_state),
                )
                prediction = self.model.apply_batch(
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
