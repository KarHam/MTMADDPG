import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg2 import MADDPG
import threading
from copy import deepcopy
import json
USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config, nodin,comm,lock,infl,task):
    ers=[]
    logger=None
    model_dir = Path('./models') / config.name/ ("node"+str(nodin))
    run_dir = model_dir 
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    with open(run_dir/'task.txt', 'w') as the_file:
        the_file.write(task)
    logger = SummaryWriter(str(log_dir))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    desc='desc.txt'
    with open( run_dir/desc, 'w') as f:
        f.write(str(infl))
        f.write(config.desc)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(task, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        #print("Episodes %i-%i of %i" % (ep_i + 1,
                                        #ep_i + 1 + config.n_rollout_threads,
                                        #config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets(comm,lock,nodin,infl)
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            ers.append(a_ep_rew)
            if((ep_i)%1000==0):
                print('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
                print(infl)
                #this should be fixed, per agent
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')
    maddpg.save(run_dir / 'model.pt')
    strek='g'+config.name+'_'+task
    
    with open(run_dir/strek, 'w') as f:
        json.dump(ers, f)
    with open('models/plotterie/'+strek, 'w') as f:
        json.dump(ers, f)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--desc", type=str)

    config = parser.parse_args()
    
    lock=threading.Lock()
    nagent=2
    par=3
    tasks=['simple_speaker_listener','simple_speaker_listener2','simple_speaker_listener3']
    process=[]
    comm=[[0 for i in range(nagent)] for j in range(par)]
    #agent, i influence him
    infl=[[[0,1,1],[0,0,0],[0,0,0]],[[0,0.25,0.25],[0.25,0,0.25],[0.25,0.25,0]]] #1 for self and 0.25 for other
    #infl=[[[1]],[[1]]]
    config.name=config.name
    while(config.name in os.listdir('models')):
        config.name=config.name+'+'
    #infl=[[[0,0],[0,0]],[[0,0],[0,0]]]
    print(comm)
    for i in range(par):
        
        process.append(threading.Thread(target=run,args=(config,i,comm,lock,infl,tasks[i])))
        process[i].start()
    process[0].join()
    print(infl)