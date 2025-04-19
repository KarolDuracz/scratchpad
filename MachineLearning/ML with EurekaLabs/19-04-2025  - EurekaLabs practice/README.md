<h2>Another TODO...</h2>
it's only idea right now, but it might be part of VM if I manage to put it together.
<br /><br />
Goal: Build an agent that executes unknown x86 (or ARM, etc.) binaries under emulation (e.g., Unicorn Engine), and uses machine learning (especially reinforcement learning) to understand the behavior of code, even when information is missing (e.g., values in registers, unknown targets of jumps, external API behavior).
<br /><br />
Right now without any description. I'm trying to find some way to speculatively execute a PE/ELF file, generally an .exe file, and based on how the code executes, the Agent would learn to understand how the code works. And instead of analyzing it through IDA, Ghirda, create an agent for it. It's more complex. But first some outline TO START SOMEWHERE.
<br /><br />

Unicorn -> https://pypi.org/project/unicorn/ - I've only just come across this, but it's so light and easy to use that it seems like the best to train an agent.
<br /><br />

needed libraries to run this

```
pip install unicorn capstone gymnasium numpy stable-baselines3 torch
```

run
```
python script.py
```


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/MachineLearning/ML%20with%20EurekaLabs/19-04-2025%20%20-%20EurekaLabs%20practice/241%20-%2019-04-2025%20-%20cd.png?raw=true)

```
# unicorn_debug_fixed.py

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from unicorn import Uc, UC_ARCH_X86, UC_MODE_32
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_EIP, UC_X86_REG_EFLAGS
from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# -----------------------------------------------------------------------
# Fixed tiny x86 snippet. Layout (addresses in comments):
#
# 0x1000: cmp  eax,5
# 0x1003: jnz  0x1013       ; if ZF==0 goto skip→mov eax,2
# 0x1009: mov  eax,1
# 0x100E: jmp  0x1018       ; skip over mov eax,2
# 0x1013: mov  eax,2
# 0x1018: nop
# -----------------------------------------------------------------------

CODE = (
    b"\x83\xF8\x05"              # cmp eax,5
    b"\x0F\x85\x0A\x00\x00\x00"  # jnz +0x0A → target = 0x1003+6 + 10 = 0x1013
    b"\xB8\x01\x00\x00\x00"      # mov eax,1
    b"\xE9\x05\x00\x00\x00"      # jmp +0x05 → target = 0x100E+5 + 5 = 0x1018
    b"\xB8\x02\x00\x00\x00"      # mov eax,2
    b"\x90"                      # nop
)

BASE_ADDRESS = 0x1000
END_ADDRESS  = BASE_ADDRESS + len(CODE)

class UnicornBranchEnv(gym.Env):
    """
    Debug‐verbose Unicorn+Gym environment.
    - Agent sees [EAX (0–10), ZF].
    - At 'jnz', chooses 0=fallthrough, 1=takebranch.
    - +1 reward for correct branch, –1 for wrong.
    """
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()
        # two actions: fallthrough or jump
        self.action_space = spaces.Discrete(2)
        # observations: EAX clipped to 10, ZF flag
        low  = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.cs = Cs(CS_ARCH_X86, CS_MODE_32)
        self._init_emulator()

    def _init_emulator(self):
        # init Unicorn
        self.mu = Uc(UC_ARCH_X86, UC_MODE_32)
        self.mu.mem_map(BASE_ADDRESS & ~0xFFF, 0x2000)
        self.mu.mem_write(BASE_ADDRESS, CODE)
        self.mu.reg_write(UC_X86_REG_EIP, BASE_ADDRESS)
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_emulator()
        # randomize EAX so sometimes ZF=1 (EAX=5), sometimes ZF=0 (EAX=6)
        start_eax = random.choice([5, 6])
        self.mu.reg_write(UC_X86_REG_EAX, start_eax)
        print(f"\n[RESET] PC=0x{BASE_ADDRESS:04X}, EAX={start_eax}")
        return self._get_obs(), {}

    def _get_obs(self):
        eax = self.mu.reg_read(UC_X86_REG_EAX) & 0xFFFFFFFF
        zf  = (self.mu.reg_read(UC_X86_REG_EFLAGS) >> 6) & 1
        return np.array([float(min(eax,10)), float(zf)], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping.")

        pc   = self.mu.reg_read(UC_X86_REG_EIP)
        code = self.mu.mem_read(pc, 16)
        insn = next(self.cs.disasm(bytes(code), pc))
        #print(f"\n[STEP] PC=0x{pc:04X}: {insn.mnemonic} {insn.op_str}")

        reward     = 0.0
        terminated = False
        truncated  = False

        if insn.mnemonic == "jnz":
            # flags from previous cmp
            eflags = self.mu.reg_read(UC_X86_REG_EFLAGS)
            zf     = (eflags >> 6) & 1
            real   = 1 if zf == 0 else 0
            reward = 1.0 if action == real else -1.0

            # compute correct absolute target
            disp   = insn.operands[0].imm
            target = insn.address + insn.size + disp
            print(f"       ZF={zf} real_branch={real}, action={action}, reward={reward}")
            print(f"       jumping → 0x{target:04X}" if action else f"       falling → 0x{pc+insn.size:04X}")

            if action:
                self.mu.reg_write(UC_X86_REG_EIP, target)
            else:
                self.mu.reg_write(UC_X86_REG_EIP, pc + insn.size)

        else:
            # actually execute the instruction to update regs & flags
            self.mu.emu_start(pc, 0, count=1)

        new_pc = self.mu.reg_read(UC_X86_REG_EIP)
        eax    = self.mu.reg_read(UC_X86_REG_EAX)
        print(f"       → next PC=0x{new_pc:04X}, EAX={eax}")

        if new_pc >= END_ADDRESS:
            self.done     = True
            terminated    = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        pc  = self.mu.reg_read(UC_X86_REG_EIP)
        eax = self.mu.reg_read(UC_X86_REG_EAX)
        print(f"[RENDER] PC=0x{pc:04X}, EAX={eax}")

"""
class TrainingLoggerCallback(BaseCallback):
    
    #Logs:
    #  • each step: observation → action → reward
   #   • each episode end: total_episode_reward
     # • each rollout end: mean value function & mean log‑probability
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        # this is called after each environment step
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)
        rewards = self.locals.get("rewards", None)
        obs     = self.locals.get("new_obs", None)

        # log the *first* (and only, since we use 1 env) env
        if obs is not None and actions is not None and rewards is not None:
            o = obs[0] if isinstance(obs, (list, tuple, np.ndarray)) else obs
            a = actions[0] if isinstance(actions, (list, tuple, np.ndarray)) else actions
            r = rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
            self.current_ep_reward += r
            print(f"[STEP] obs={o}, action={a}, reward={r:.2f}")

        # check for end of episode
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                total_r = ep["r"]
                length  = ep["l"]
                self.episode_rewards.append(total_r)
                print(f"[EPISODE] #{len(self.episode_rewards)} finished — "
                      f"length={length}, total_reward={total_r:.2f}")
                self.current_ep_reward = 0.0

        return True

    def _on_rollout_end(self) -> None:
        # called at the end of each rollout (n_steps)
        values   = self.locals.get("values")
        log_probs= self.locals.get("log_probs")
        if values is not None and log_probs is not None:
            mean_val = float(np.mean(values))
            mean_lp  = float(np.mean(log_probs))
            print(f"[ROLLOUT] mean_value={mean_val:.3f}, mean_logprob={mean_lp:.3f}")
        return None
"""

class TrainingLoggerCallback(BaseCallback):
    """
    Logs:
      • each step: obs → action → reward
      • each episode end: total_episode_reward
      • each rollout end: mean value function & mean log‑probability
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)
        rewards = self.locals.get("rewards", None)
        obs     = self.locals.get("new_obs", None)

        if obs is not None and actions is not None and rewards is not None:
            # handle vector vs single env
            o = obs[0] if isinstance(obs, (list, tuple, np.ndarray)) else obs
            a = actions[0] if isinstance(actions, (list, tuple, np.ndarray)) else actions
            r = rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
            self.current_ep_reward += r
            #print(f"[STEP] obs={o}, action={a}, reward={r:.2f}")

        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                total_r = ep["r"]
                length  = ep["l"]
                self.episode_rewards.append(total_r)
                print(f"[EPISODE] #{len(self.episode_rewards)} finished — "
                      f"length={length}, total_reward={total_r:.2f}")
                self.current_ep_reward = 0.0

        return True

    def _on_rollout_end(self) -> None:
        # called at end of each rollout
        values    = self.locals.get("values")
        log_probs = self.locals.get("log_probs")

        def tensor_mean(x):
            # x might be a torch tensor or numpy array
            if hasattr(x, "mean") and hasattr(x, "item"):
                # torch Tensor
                return float(x.mean().item())
            else:
                # fallback to numpy
                return float(np.mean(x))

        if values is not None and log_probs is not None:
            mean_val = tensor_mean(values)
            mean_lp  = tensor_mean(log_probs)
            print(f"[ROLLOUT] mean_value={mean_val:.3f}, mean_logprob={mean_lp:.3f}")
        return None

"""
if __name__ == "__main__":
    env = UnicornBranchEnv()

    # Train PPO for a short demo
    print("\n=== START TRAINING ===")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20_000)
    model.save("ppo_debug_fixed")

    # Single test episode
    print("\n=== TEST EPISODE ===")
    obs, _       = env.reset()
    done         = False
    total_reward = 0.0

    while not done:
        action, _         = model.predict(obs, deterministic=True)
        obs, reward, t, tr, _ = env.step(action)
        total_reward     += reward
        done              = t or tr

    print(f"\n[RESULT] Total reward = {total_reward:.1f}")
"""

if __name__ == "__main__":
    # 1) wrap your env in a Monitor to collect episode info
    base_env = UnicornBranchEnv()
    env = Monitor(base_env)

    # 2) create PPO with verbose=0 (we handle printing ourselves)
    model = PPO("MlpPolicy", env, verbose=0)

    # 3) train with our callback
    print("\n=== START TRAINING WITH LOGGING ===")
    callback = TrainingLoggerCallback()
    model.learn(total_timesteps=2_000, callback=callback)
    model.save("ppo_debug_fixed_stats")

    # 4) test one episode (you still get the per‑step prints from your env)
    print("\n=== TEST EPISODE ===")
    obs, _       = env.reset()
    done         = False
    total_reward = 0.0

    while not done:
        action, _         = model.predict(obs, deterministic=True)
        obs, reward, t, tr, info = env.step(action)
        total_reward     += reward
        done              = t or tr

    print(f"\n[RESULT] Test episode total reward = {total_reward:.1f}")
```
